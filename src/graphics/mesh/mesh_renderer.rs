use crate::{
    graphics::{
        gi::{GlobalIlluminationPipeline, LIGHT_TEX_FMT},
        manager::MeshId,
        material::Material,
        renderer::{DEFAULT_MULTISAMPLE_STATE, DEPTH_FORMAT, SWAPCHAIN_FORMAT},
        util::GpuMat4,
        Camera, GraphicsManager,
    },
    math as m,
};

use std::mem::size_of;
use zerocopy::{AsBytes, FromBytes};

pub struct MeshRenderer {
    // depth prepass to prevent overdraw
    depth_pipeline: wgpu::RenderPipeline,
    // light pass drawing light emitters and occluders to a different target
    emissive_pipeline: wgpu::RenderPipeline,
    // pipeline actually drawing the meshes
    main_pipeline: wgpu::RenderPipeline,
    instance_unif_buf: wgpu::Buffer,
    instance_unif_bind_group_layout: wgpu::BindGroupLayout,
    instance_unif_bind_group: wgpu::BindGroup,
    instance_capacity: usize,
    meshes_sorted: Vec<(MeshId, Option<m::Pose>)>,
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
    pub(crate) fn new(gi_pl: &GlobalIlluminationPipeline) -> Self {
        let device = crate::Renderer::device();

        let depth_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/depth_and_emissive.wgsl"));
        let main_shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/mesh.wgsl"));

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
                    offset: 4 * 3,
                    shader_location: 1,
                },
                // normal
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 4 * 3 + 4 * 2,
                    shader_location: 2,
                },
                // tangent
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 4 * 3 + 4 * 2 + 4 * 3,
                    shader_location: 3,
                },
            ],
        };

        //
        // pipelines
        //

        let depth_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("depth prepass"),
            bind_group_layouts: &[
                crate::Camera::bind_group_layout(),
                Material::bind_group_layout(),
                &instance_unif_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("depth prepass"),
            layout: Some(&depth_pl_layout),
            vertex: wgpu::VertexState {
                module: &depth_shader,
                entry_point: "vs_main",
                buffers: &[vertex_buffers.clone()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &depth_shader,
                entry_point: "fs_depth",
                targets: &[],
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
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    ..Default::default()
                },
            }),
            multisample: DEFAULT_MULTISAMPLE_STATE,
            multiview: None,
        });

        let emissive_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("emissive"),
            layout: Some(&depth_pl_layout),
            vertex: wgpu::VertexState {
                module: &depth_shader,
                entry_point: "vs_main",
                buffers: &[vertex_buffers.clone()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &depth_shader,
                entry_point: "fs_emissive",
                targets: &[Some(LIGHT_TEX_FMT.into()), Some(LIGHT_TEX_FMT.into())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            // when drawing lights we draw to a different, lower-res render target
            // without depth testing or multisampling
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let main_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh"),
            bind_group_layouts: &[
                crate::Camera::bind_group_layout(),
                &gi_pl.bind_group_layouts.render,
                Material::bind_group_layout(),
                &instance_unif_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let main_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh"),
            layout: Some(&main_pl_layout),
            vertex: wgpu::VertexState {
                module: &main_shader,
                entry_point: "vs_main",
                buffers: &[vertex_buffers],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &main_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: SWAPCHAIN_FORMAT,
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
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: DEFAULT_MULTISAMPLE_STATE,
            multiview: None,
        });

        Self {
            depth_pipeline,
            emissive_pipeline,
            main_pipeline,
            instance_unif_buf,
            instance_unif_bind_group_layout,
            instance_unif_bind_group,
            instance_capacity: 1,
            meshes_sorted: Vec::new(),
        }
    }

    /// Sort meshes by depth and upload their information to the gpu.
    pub(crate) fn prepare(&mut self, manager: &mut GraphicsManager, world: &mut hecs::World) {
        let device = crate::Renderer::device();
        let queue = crate::Renderer::queue();

        self.meshes_sorted.clear();
        self.meshes_sorted.extend(
            world
                .query_mut::<(&MeshId, Option<&m::Pose>)>()
                .into_iter()
                .map(|(_, (id, pose))| (*id, pose.copied())),
        );
        // sort in z order for transparency and efficient depth prepass.
        // the z order of meshes very rarely changes,
        // so there's some room for perf gains here by caching the order,
        // but it's a little finicky to do well.
        // prefer to profile before doing that
        self.meshes_sorted.sort_by(|(_, pose_a), (_, pose_b)| {
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
        for (mesh_id, pose) in &self.meshes_sorted {
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
    }

    /// Draw meshes into the depth buffer.
    pub(crate) fn depth_pass<'pass>(
        &'pass mut self,
        pass: &mut wgpu::RenderPass<'pass>,
        manager: &'pass mut GraphicsManager,
        camera: &'pass Camera,
    ) {
        // depth prepass in forward z order (+z is away from camera),
        // drawing closest things first to maximize depth test discards
        pass.set_pipeline(&self.depth_pipeline);
        pass.set_bind_group(0, &camera.bind_group, &[]);

        for (idx, (mesh_id, _)) in self.meshes_sorted.iter().enumerate() {
            self.draw_mesh(pass, manager, idx, mesh_id, PassId::Depth);
        }
    }

    /// Draw meshes with emissive / occlusive materials to the light texture
    /// for global illumination (see gi.rs)
    pub(crate) fn emissive_pass<'pass>(
        &'pass self,
        pass: &mut wgpu::RenderPass<'pass>,
        manager: &'pass mut GraphicsManager,
        camera: &'pass Camera,
    ) {
        pass.set_pipeline(&self.emissive_pipeline);
        pass.set_bind_group(0, &camera.bind_group, &[]);
        for (idx, (mesh_id, _)) in self.meshes_sorted.iter().enumerate() {
            self.draw_mesh(pass, manager, idx, mesh_id, PassId::Emissive);
        }
    }

    pub(crate) fn draw_pass<'pass>(
        &'pass mut self,
        pass: &mut wgpu::RenderPass<'pass>,
        manager: &'pass mut GraphicsManager,
        camera: &'pass Camera,
        gi_pl: &'pass GlobalIlluminationPipeline,
    ) {
        // full render in reverse z order for transparency

        pass.set_pipeline(&self.main_pipeline);
        pass.set_bind_group(0, &camera.bind_group, &[]);
        pass.set_bind_group(1, &gi_pl.bind_groups.render, &[]);

        for (idx, (mesh_id, _)) in self.meshes_sorted.iter().enumerate().rev() {
            self.draw_mesh(pass, manager, idx, mesh_id, PassId::Main);
        }
    }

    fn draw_mesh<'pass>(
        &'pass self,
        pass: &mut wgpu::RenderPass<'pass>,
        manager: &'pass GraphicsManager,
        idx: usize,
        mesh_id: &MeshId,
        pass_id: PassId,
    ) {
        let Some(mesh) = manager.get_mesh(mesh_id) else {
            return;
        };

        let material = manager.get_mesh_material(mesh_id);
        if matches!(pass_id, PassId::Emissive) && !material.participates_in_lighting {
            return;
        }

        let bg_offset = match pass_id {
            PassId::Main => 1,
            _ => 0,
        };
        pass.set_bind_group(1 + bg_offset, &material.bind_group, &[]);
        pass.set_bind_group(
            2 + bg_offset,
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

#[derive(Clone, Copy)]
enum PassId {
    Depth,
    Emissive,
    Main,
}
