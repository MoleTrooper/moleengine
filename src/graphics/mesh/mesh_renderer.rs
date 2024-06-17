use crate::{
    graphics::{
        light::LightBuffers,
        manager::MeshId,
        material::Material,
        renderer::{DEPTH_FORMAT, SWAPCHAIN_FORMAT},
        util::GpuMat4,
        Camera, GraphicsManager,
    },
    math as m,
};

use std::mem::size_of;
use zerocopy::{AsBytes, FromBytes};

pub struct MeshRenderer {
    pipeline: wgpu::RenderPipeline,
    // same for instance uniforms
    instance_unif_buf: wgpu::Buffer,
    instance_unif_bind_group_layout: wgpu::BindGroupLayout,
    instance_unif_bind_group: wgpu::BindGroup,
    instance_capacity: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, AsBytes, FromBytes)]
struct InstanceUniforms {
    // space could be saved here by packing the model matrix into three row vectors,
    // but alignment with dynamic offsets requires a minimum of 64 bytes
    // and we don't currently have anything but the model matrix here
    // so it might as well be the full 4x4 matrix
    model: GpuMat4,
}

impl MeshRenderer {
    pub(crate) fn new(light_bufs: &LightBuffers) -> Self {
        let device = crate::Renderer::device();

        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/mesh.wgsl"));

        // instance uniforms bind group

        let instance_unif_buf = device.create_buffer(&wgpu::BufferDescriptor {
            size: size_of::<InstanceUniforms>() as _,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            label: Some("mesh instance uniforms"),
            mapped_at_creation: false,
        });

        let instance_unif_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mesh instance uniforms"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(size_of::<InstanceUniforms>() as _),
                    },
                    count: None,
                }],
            });

        let instance_unif_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh instance uniforms"),
            layout: &instance_unif_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: instance_unif_buf.as_entire_binding(),
            }],
        });

        // vertex layouts

        let vertex_buffers = wgpu::VertexBufferLayout {
            array_stride: size_of::<super::Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                // texture coordinates
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 4 * 4,
                    shader_location: 1,
                },
                // normal
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 4 * 4 + 4 * 4,
                    shader_location: 2,
                },
                // tangent
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 4 * 4 + 4 * 4 + 4 * 4,
                    shader_location: 3,
                },
            ],
        };

        //
        // pipeline
        //

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh"),
            bind_group_layouts: &[
                crate::Camera::bind_group_layout(),
                &light_bufs.bind_group_layout,
                Material::bind_group_layout(),
                &instance_unif_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_buffers],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: SWAPCHAIN_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline,
            instance_unif_buf,
            instance_unif_bind_group_layout,
            instance_unif_bind_group,
            instance_capacity: 1,
        }
    }

    /// Draw all the meshes in the world.
    pub(crate) fn draw<'pass>(
        &'pass mut self,
        pass: &mut wgpu::RenderPass<'pass>,
        manager: &'pass mut GraphicsManager,
        world: &mut hecs::World,
        camera: &'pass Camera,
        light_bufs: &'pass LightBuffers,
    ) {
        let device = crate::Renderer::device();
        let queue = crate::Renderer::queue();

        let mut meshes_in_world: Vec<(MeshId, Option<m::Pose>)> = world
            .query_mut::<(&MeshId, Option<&m::Pose>)>()
            .into_iter()
            .map(|(_, (id, pose))| (*id, pose.copied()))
            .collect();
        // sort in z order for transparency and efficient depth prepass.
        // the z order of meshes very rarely changes,
        // so there's some room for perf gains here by caching the order,
        // but it's a little finicky to do well.
        // prefer to profile before doing that
        meshes_in_world.sort_by(|(_, pose_a), (_, pose_b)| {
            let z_a = pose_a.map(|p| p.translation.z).unwrap_or(0.);
            let z_b = pose_b.map(|p| p.translation.z).unwrap_or(0.);
            z_a.total_cmp(&z_b)
        });

        //
        // gather uniforms
        //

        // collect all instance uniforms into a big buffer;
        // we'll use dynamic offsets to bind them
        let mut instance_unifs = Vec::new();
        for (mesh_id, pose) in &meshes_in_world {
            let Some(mesh) = manager.get_mesh_mut(mesh_id) else {
                continue;
            };

            let model = match pose {
                Some(entity_pose) => *entity_pose * mesh.offset,
                None => mesh.offset,
            }
            .into_homogeneous_matrix();

            instance_unifs.push(InstanceUniforms {
                model: model.into(),
            });
        }

        if instance_unifs.len() > self.instance_capacity {
            self.instance_unif_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mesh instance uniforms"),
                size: (size_of::<InstanceUniforms>() * instance_unifs.len()) as _,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_unif_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mesh instance uniforms"),
                layout: &self.instance_unif_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.instance_unif_buf,
                        offset: 0,
                        // size set manually instead of using as_entire_binding
                        // because we're using dynamic offsets
                        size: wgpu::BufferSize::new(size_of::<InstanceUniforms>() as _),
                    }),
                }],
            });
        }

        queue.write_buffer(&self.instance_unif_buf, 0, instance_unifs.as_bytes());

        //
        // render
        //

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &camera.bind_group, &[]);
        pass.set_bind_group(1, &light_bufs.bind_group, &[]);

        // reverse z order - +z is away from camera
        for (idx, (mesh_id, _)) in meshes_in_world.iter().enumerate().rev() {
            let Some(mesh) = manager.get_mesh(mesh_id) else {
                continue;
            };

            let material = manager.get_mesh_material(mesh_id);
            pass.set_bind_group(2, &material.bind_group, &[]);
            pass.set_bind_group(
                3,
                &self.instance_unif_bind_group,
                &[(idx * size_of::<InstanceUniforms>()) as u32],
            );

            if let Some(target) = mesh_id
                .skin
                .and_then(|skin_id| manager.skins.get(skin_id))
                .and_then(|skin| skin.compute_res.as_ref())
            {
                pass.set_vertex_buffer(0, target.vertex_buf.slice(..));
            } else {
                pass.set_vertex_buffer(0, mesh.gpu_data.vertex_buf.slice(..));
            }
            pass.set_index_buffer(mesh.gpu_data.index_buf.slice(..), wgpu::IndexFormat::Uint16);

            pass.draw_indexed(0..mesh.gpu_data.idx_count, 0, 0..1);
        }
    }
}
