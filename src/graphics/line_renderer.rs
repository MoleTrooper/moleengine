use crate::{
    graphics::{
        renderer::{DEFAULT_MULTISAMPLE_STATE, DEPTH_FORMAT},
        util::DynamicBuffer,
    },
    math::uv,
    MaterialId,
};

use std::borrow::Cow;
use zerocopy::{AsBytes, FromBytes};

#[derive(Clone, Copy, Debug)]
pub struct LineVertex {
    pub position: uv::Vec3,
    pub width: f32,
}

pub struct LineStrip {
    instance_buf: DynamicBuffer,
    point_count: u32,
    material_id: Option<MaterialId>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, AsBytes, FromBytes)]
struct GpuVertex {
    position: [f32; 3],
    width: f32,
}

impl From<LineVertex> for GpuVertex {
    fn from(v: LineVertex) -> Self {
        Self {
            position: v.position.into(),
            width: v.width,
        }
    }
}

impl LineStrip {
    /// Create a new line strip.
    ///
    /// If a material id is not given, the default material is used.
    pub fn new(vertices: &[LineVertex], material_id: Option<MaterialId>) -> Self {
        let point_count = vertices.len() as u32;
        assert!(
            point_count >= 2,
            "A line strip must have at least two points"
        );
        let mut instance_buf = DynamicBuffer::new(None, wgpu::BufferUsages::VERTEX);
        let mut gpu_vertices: Vec<GpuVertex> =
            vertices.iter().copied().map(GpuVertex::from).collect();
        // duplicate the second to last vertex
        // so that we can draw both caps with the same pipeline
        // (otherwise we'd need a second one with step order reversed,
        // or some other tricky business)
        gpu_vertices.push(*gpu_vertices.iter().rev().nth(1).unwrap());
        instance_buf.write(&gpu_vertices);
        Self {
            instance_buf,
            point_count,
            material_id,
        }
    }

    /// Rewrite the vertices of this line.
    ///
    /// This is more efficient than dropping and making a new one,
    /// especially if the number of vertices does not change.
    pub fn overwrite(&mut self, vertices: &[LineVertex]) {
        self.point_count = vertices.len() as u32;
        assert!(
            self.point_count >= 2,
            "A line strip must have at least two points"
        );
        let mut gpu_vertices: Vec<GpuVertex> =
            vertices.iter().copied().map(GpuVertex::from).collect();
        gpu_vertices.push(*gpu_vertices.iter().rev().nth(1).unwrap());
        self.instance_buf.write(&gpu_vertices);
    }
}

/// A versatile instanced line renderer.
///
/// Loosely based on [this blog post by Rye Terrell]
/// (https://wwwtyro.net/2019/11/18/instanced-lines.html)
/// and [this one](https://wwwtyro.net/2021/10/01/instanced-lines-part-2.html).
pub struct LineRenderer {
    // pipelines for different instance step modes and shaders
    pipelines: Pipelines,
    // geometry for segments and joins
    primitives: Primitives,
}

/// A collection of all the pipelines with different step modes and shaders.
struct Pipelines {
    // line segments are rendered one half at a time in two passes,
    // once iterating forward and once backward
    seg_forward: wgpu::RenderPipeline,
    seg_backward: wgpu::RenderPipeline,
    cap: wgpu::RenderPipeline,
}

/// A vertex in the instance geometry.
type InstanceVertex = [f32; 3];

/// A collection of all instance primitives we need
/// for line segments, caps, and joins.
struct Primitives {
    semisegment: InstanceGeometry,
    cap_segment: InstanceGeometry,
}

impl Primitives {
    /// Generate vertex and index buffers for line segments, joins and caps.
    fn generate_instance_geometry() -> Self {
        let device = crate::Renderer::device();

        // a line segment is a rectangle
        // with coordinates chosen so that it's easy to transform
        // using the difference between two points and a thickness value,
        // plus some special geometry for a round join at one end
        let semisegment = InstanceGeometry::upload(
            device,
            "line segment",
            &[
                // segment
                [0., 0.5, 0.],
                [0., -0.5, 0.],
                [0.5, -0.5, 0.],
                [0.5, 0.5, 0.],
                // join geometry is expressed with the z coordinates
                [0., 0.5, 1.],
                [0., 0.5, 2.],
            ],
            &[0, 1, 2, 0, 2, 3, 1, 4, 0, 1, 5, 4],
        );

        let cap_segment = InstanceGeometry::upload(
            device,
            "cap segment",
            &[
                [0., 0.5, 0.],
                [0., -0.5, 0.],
                [0.5, -0.5, 0.],
                [0.5, 0.5, 0.],
            ],
            &[0, 1, 2, 0, 2, 3],
        );

        Self {
            semisegment,
            cap_segment,
        }
    }
}

impl LineRenderer {
    pub(crate) fn new() -> Self {
        let device = crate::Renderer::device();

        let label = Some("line");

        let segment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "shaders/line_segment.wgsl"
            ))),
        });
        let cap_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/line_cap.wgsl"))),
        });

        // pipeline

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label,
            bind_group_layouts: &[
                crate::Camera::bind_group_layout(),
                crate::Material::bind_group_layout(),
            ],
            push_constant_ranges: &[],
        });

        /// different rates of stepping through the instance buffer and different shaders
        /// are needed for different parts of drawing the lines
        enum InstanceStepMode {
            SegmentsForward,
            SegmentsBackward,
            Caps,
        }

        let pipeline = |mode: InstanceStepMode| {
            use InstanceStepMode::*;
            let label = Some(match mode {
                SegmentsForward => "segments forward",
                SegmentsBackward => "segments backward",
                Caps => "caps",
            });
            let instance_attributes = match mode {
                SegmentsForward => [
                    // previous point
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 0,
                        shader_location: 1,
                    },
                    // current point
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 4 * 4,
                        shader_location: 2,
                    },
                    // next point
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 4 * 4 * 2,
                        shader_location: 3,
                    },
                ]
                .as_slice(),
                // order reversed for backward
                SegmentsBackward => [
                    // previous point
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 0,
                        shader_location: 3,
                    },
                    // current point
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 4 * 4,
                        shader_location: 2,
                    },
                    // next point
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 4 * 4 * 2,
                        shader_location: 1,
                    },
                ]
                .as_slice(),
                Caps => [
                    // current point
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 0,
                        shader_location: 1,
                    },
                    // next point
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 4 * 4,
                        shader_location: 2,
                    },
                ]
                .as_slice(),
            };
            let module = match mode {
                SegmentsBackward | SegmentsForward => &segment_shader,
                Caps => &cap_shader,
            };

            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: "vs_main",
                    buffers: &[
                        // vertices of a single line segment instance
                        wgpu::VertexBufferLayout {
                            // three floats for (x,y,z) coordinates
                            array_stride: 3 * 4,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[
                                // position
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x3,
                                    offset: 0,
                                    shader_location: 0,
                                },
                            ],
                        },
                        // instance buffer containing start and end points of line segments
                        wgpu::VertexBufferLayout {
                            // always stepping a point at a time
                            array_stride: 4 * 4,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: instance_attributes,
                        },
                    ],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &segment_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: super::renderer::SWAPCHAIN_FORMAT,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::COLOR,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: DEFAULT_MULTISAMPLE_STATE,
                multiview: None,
            })
        };

        Self {
            pipelines: Pipelines {
                seg_forward: pipeline(InstanceStepMode::SegmentsForward),
                seg_backward: pipeline(InstanceStepMode::SegmentsBackward),
                cap: pipeline(InstanceStepMode::Caps),
            },
            primitives: Primitives::generate_instance_geometry(),
        }
    }

    pub fn draw<'pass>(
        &'pass self,
        pass: &mut wgpu::RenderPass<'pass>,
        manager: &'pass crate::GraphicsManager,
        camera: &'pass crate::Camera,
        line: &'pass LineStrip,
    ) {
        let material = if let Some(mid) = line.material_id {
            manager.get_material(mid)
        } else {
            crate::Material::get_default()
        };
        pass.set_bind_group(0, &camera.bind_group, &[]);
        pass.set_bind_group(1, &material.bind_group, &[]);

        let intermediate_segment_count = line.point_count - 2;

        // segments

        let idx_range = self.primitives.semisegment.bind(pass);
        pass.set_vertex_buffer(1, line.instance_buf.slice());

        pass.set_pipeline(&self.pipelines.seg_forward);
        pass.draw_indexed(idx_range.clone(), 0, 0..intermediate_segment_count);
        pass.set_pipeline(&self.pipelines.seg_backward);
        pass.draw_indexed(idx_range, 0, 0..intermediate_segment_count);

        // caps

        let idx_range = self.primitives.cap_segment.bind(pass);
        pass.set_pipeline(&self.pipelines.cap);
        pass.draw_indexed(idx_range.clone(), 0, 0..1);
        // second to last point in the line has been duplicated for this
        pass.draw_indexed(idx_range, 0, line.point_count - 1..line.point_count);
    }
}

//
// utility types
//

/// Vertex and index buffer to hold a mesh instance.
struct InstanceGeometry {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
}

impl InstanceGeometry {
    /// Upload vertices and indices to the GPU.
    fn upload(
        device: &wgpu::Device,
        label: &str,
        vertices: &[InstanceVertex],
        indices: &[u16],
    ) -> Self {
        use wgpu::util::DeviceExt;
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: vertices.as_bytes(),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: indices.as_bytes(),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buf,
            index_buf,
            index_count: indices.len() as u32,
        }
    }

    /// Bind this instance's geometry to vertex buffer 0.
    /// Returns the index range to draw with for extra convenience.
    fn bind<'pass>(&'pass self, pass: &mut wgpu::RenderPass<'pass>) -> std::ops::Range<u32> {
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
        0..self.index_count
    }
}
