use std::mem::size_of;

use wgpu::util::DeviceExt;
use zerocopy::{AsBytes, FromBytes};

use crate::math::uv;
use wgpu_profiler as wp;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightingQualityConfig {
    /// Distance between light probes measured in screenspace pixels.
    /// Lower is better quality and more expensive.
    pub probe_interval: f32,
    /// Whether or not to run raymarching for the final cascade.
    /// Significantly reduces quality around light emitter edges
    /// but improves performance especially when probe spacing is high.
    /// Mainly useful for squeezing out a bit of extra performance
    /// on extremely low settings.
    pub skip_final_cascade: bool,
}

impl LightingQualityConfig {
    pub const ULTRA: Self = Self {
        probe_interval: 1.,
        skip_final_cascade: false,
    };
    pub const HIGH: Self = Self {
        probe_interval: 2.,
        skip_final_cascade: false,
    };
    pub const MEDIUM: Self = Self {
        probe_interval: 4.,
        skip_final_cascade: false,
    };
    pub const LOW: Self = Self {
        probe_interval: 6.,
        skip_final_cascade: false,
    };
    pub const LOWEST: Self = Self {
        probe_interval: 8.,
        skip_final_cascade: true,
    };

    // Get the range of a c0 probe, which is half the diagonal of a square between probes
    #[inline]
    fn probe_range(self) -> f32 {
        (2. * self.probe_interval).sqrt()
    }
}

impl Default for LightingQualityConfig {
    fn default() -> Self {
        Self::HIGH
    }
}

const CASCADE_TEX_FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Workgroups for compute shaders are arranged in squares of this size.
const TILE_SIZE: u32 = 16;

pub(crate) struct GlobalIlluminationPipeline {
    quality_conf: LightingQualityConfig,

    pipelines: Pipelines,
    pub(super) textures: Textures,
    pub(super) bind_group_layouts: BindGroupLayouts,
    pub(super) bind_groups: BindGroups,
    buffers: Buffers,
    // sampler for interpolating nearest probes during rendering
    // (can't use it when combining cascades
    // because sampling is not allowed in compute shaders)
    bilinear_samp: wgpu::Sampler,
    light_tex_size: (u32, u32),
    last_screen_size: winit::dpi::PhysicalSize<u32>,
    cascade_count: usize,
    probe_count: [u32; 2],
}

//
// gpu resources
//

struct Pipelines {
    // generates a mip chain from the light texture
    // that we use as a quadtree-like acceleration structure for raymarching
    light_mip: wgpu::ComputePipeline,
    // actual radiance cascade computation
    cascade: wgpu::RenderPipeline,
}

pub(super) struct BindGroupLayouts {
    light: wgpu::BindGroupLayout,
    light_mip: wgpu::BindGroupLayout,
    cascade: wgpu::BindGroupLayout,
    pub(super) render: wgpu::BindGroupLayout,
}

pub(super) struct BindGroups {
    // binds the light texture for the cascade compute step
    light_full: wgpu::BindGroup,
    light_mips: Vec<wgpu::BindGroup>,
    // back and forth bind groups for cascades
    cascades: [wgpu::BindGroup; 2],
    pub(super) render: wgpu::BindGroup,
}

pub(super) struct Textures {
    // light emitters and occluders are drawn into this by MeshRenderer
    // (one texture with LIGHT_MIP_COUNT mip levels,
    // the rest of the levels are used to accelerate raymarching)
    // this view has all the mip levels
    light_full: wgpu::TextureView,
    // and these are one mip level each, used to generate the mip chain
    pub(super) light_mips: Vec<wgpu::TextureView>,
    // other cascades alternate between two textures of the same size
    cascades: [wgpu::TextureView; 2],
}

struct Buffers {
    frame_params: wgpu::Buffer,
    cascade_params: wgpu::Buffer,
    render_params: wgpu::Buffer,
}

/// Resources that need to be resized when screen size changes
struct ResizeResults {
    light_tex_size: (u32, u32),
    textures: Textures,
    cascade_params: Vec<CascadeParams>,
    render_params: RenderParams,
}

//
// gpu uniform types
//

#[repr(C)]
#[derive(Clone, Copy, Debug, AsBytes, FromBytes)]
struct FrameParams {
    pixel_size_10cm: f32,
}

/// Uniform parameters for the cascade computation
#[repr(C)]
#[derive(Clone, Copy, Debug, AsBytes, FromBytes)]
struct CascadeParams {
    level: u32,
    level_count: u32,
    probe_count: [u32; 2],
    rays_per_probe: u32,
    // spacing and range in the light texture's pixel space,
    // not screen space
    linear_spacing: f32,
    range_start: f32,
    range_length: f32,
    // padding to reach the minimum dynamic offset alignment
    _pad: [u32; 8],
}

impl CascadeParams {
    /// Compute the parameters for a given cascade level.
    ///
    /// On each level, the number of probes is reduced by a factor of 4
    /// (halved in both dimensions)
    /// and the number of rays per probe is increased by a factor of 4.
    fn for_level(
        level: u32,
        cascade_count: u32,
        probe_count_c0: [u32; 2],
        config: LightingQualityConfig,
    ) -> Self {
        let spacing_c0 = config.probe_interval;
        let range_c0 = config.probe_range();

        let level_exp2 = 1 << level;
        let level_exp4 = level_exp2 * level_exp2;
        CascadeParams {
            level,
            level_count: cascade_count,
            probe_count: probe_count_c0.map(|c| c / level_exp2),
            rays_per_probe: level_exp4,
            linear_spacing: spacing_c0 * level_exp2 as f32,
            // each range is 4 times larger than the previous,
            // and starts at the previous one's end,
            // hence the start distance is the sum of a geometric sequence
            range_start: range_c0 * (1. - level_exp4 as f32) / (1. - 4.),
            range_length: range_c0 * level_exp4 as f32,
            _pad: [0; 8],
        }
    }
}

/// Parameters for the mesh renderer to use the cascade results
#[repr(C)]
#[derive(Clone, Copy, Debug, AsBytes, FromBytes)]
struct RenderParams {
    probe_spacing: f32,
    probe_range: f32,
    probe_count: [u32; 2],
    // this is actually a bool
    // but that doesn't work with AsBytes/FromBytes
    skip_raymarch: u32,
    _pad: u32,
}

impl GlobalIlluminationPipeline {
    pub fn new(quality_conf: LightingQualityConfig) -> Self {
        let device = crate::Renderer::device();

        let bind_group_layouts = Self::create_bind_group_layouts();

        let window_size = crate::Renderer::window().inner_size();
        let resizables = Self::create_resizables(window_size.into(), quality_conf);
        let cascade_count = resizables.cascade_params.len();

        let bilinear_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        use wgpu::util::DeviceExt;
        let buffers = Buffers {
            frame_params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cascade per-frame params"),
                size: size_of::<FrameParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            cascade_params: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cascade compute params"),
                contents: resizables.cascade_params.as_bytes(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }),
            render_params: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cascade render params"),
                contents: resizables.render_params.as_bytes(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }),
        };

        let bind_groups = Self::create_bind_groups(
            cascade_count,
            &bind_group_layouts,
            &resizables.textures,
            &buffers,
            &bilinear_samp,
        );

        let pipelines = Self::create_pipelines(&bind_group_layouts);

        Self {
            quality_conf,
            pipelines,
            textures: resizables.textures,
            bind_group_layouts,
            bind_groups,
            buffers,
            bilinear_samp,
            light_tex_size: resizables.light_tex_size,
            last_screen_size: window_size,
            cascade_count,
            probe_count: resizables.cascade_params[0].probe_count,
        }
    }

    fn create_pipelines(bg_layouts: &BindGroupLayouts) -> Pipelines {
        let device = crate::Renderer::device();

        let mip_shader =
            device.create_shader_module(wgpu::include_wgsl!("./shaders/light_mip.wgsl"));

        let mip_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("light mip"),
            bind_group_layouts: &[&bg_layouts.light_mip],
            push_constant_ranges: &[],
        });

        let light_mip = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("light mip"),
            module: &mip_shader,
            entry_point: "main",
            layout: Some(&mip_layout),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let casc_shader =
            device.create_shader_module(wgpu::include_wgsl!("./shaders/radiance_cascades.wgsl"));

        let cascade_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("radiance cascades"),
            bind_group_layouts: &[&bg_layouts.light, &bg_layouts.cascade],
            push_constant_ranges: &[],
        });

        let cascade = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("radiance cascades"),
            layout: Some(&cascade_layout),
            vertex: wgpu::VertexState {
                module: &casc_shader,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &casc_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: CASCADE_TEX_FMT,
                    blend: None,
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Pipelines { light_mip, cascade }
    }

    fn create_bind_group_layouts() -> BindGroupLayouts {
        let device = crate::Renderer::device();

        let uniform_buf = |binding: u32,
                           size: usize,
                           has_dynamic_offset: bool,
                           visibility: wgpu::ShaderStages| {
            wgpu::BindGroupLayoutEntry {
                binding,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset,
                    min_binding_size: wgpu::BufferSize::new(size as u64),
                },
                visibility,
                count: None,
            }
        };

        let float_tex = |binding: u32, visibility: wgpu::ShaderStages, filterable: bool| {
            wgpu::BindGroupLayoutEntry {
                binding,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                visibility,
                count: None,
            }
        };

        let filtering_sampler =
            |binding: u32, visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
                binding,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                visibility,
                count: None,
            };

        let write_storage =
            |binding: u32, format: wgpu::TextureFormat| wgpu::BindGroupLayoutEntry {
                binding,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
            };

        use wgpu::ShaderStages as S;

        let light = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light texture for cascades"),
            entries: &[
                float_tex(0, S::FRAGMENT, false),
                uniform_buf(1, size_of::<FrameParams>(), false, S::FRAGMENT),
            ],
        });

        let light_mip = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light mip"),
            entries: &[
                float_tex(0, S::COMPUTE, false),
                write_storage(1, wgpu::TextureFormat::Rgba8Unorm),
            ],
        });

        let cascade = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cascade"),
            entries: &[
                uniform_buf(0, size_of::<CascadeParams>(), true, S::FRAGMENT),
                float_tex(1, S::FRAGMENT, true),
                filtering_sampler(2, S::FRAGMENT),
            ],
        });

        let render = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cascade render"),
            entries: &[
                uniform_buf(0, size_of::<RenderParams>(), false, S::FRAGMENT),
                float_tex(1, S::FRAGMENT, true),
                float_tex(2, S::FRAGMENT, true),
                filtering_sampler(3, S::FRAGMENT),
            ],
        });

        BindGroupLayouts {
            light,
            light_mip,
            cascade,
            render,
        }
    }

    fn create_bind_groups(
        cascade_count: usize,
        layouts: &BindGroupLayouts,
        tex: &Textures,
        buffers: &Buffers,
        bilinear_samp: &wgpu::Sampler,
    ) -> BindGroups {
        let device = crate::Renderer::device();

        let light = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cascade light"),
            layout: &layouts.light,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tex.light_full),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.frame_params.as_entire_binding(),
                },
            ],
        });

        let light_mips = (0..cascade_count - 1)
            .map(|mip_idx| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("light mip"),
                    layout: &layouts.light_mip,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&tex.light_mips[mip_idx]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &tex.light_mips[mip_idx + 1],
                            ),
                        },
                    ],
                })
            })
            .collect();

        let cascade = |read: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cascade"),
                layout: &layouts.cascade,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &buffers.cascade_params,
                            offset: 0,
                            size: wgpu::BufferSize::new(size_of::<CascadeParams>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(read),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(bilinear_samp),
                    },
                ],
            })
        };

        let render = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cascade render"),
            layout: &layouts.render,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.render_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&tex.light_full),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&tex.cascades[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(bilinear_samp),
                },
            ],
        });

        BindGroups {
            light_full: light,
            light_mips,
            cascades: [cascade(&tex.cascades[0]), cascade(&tex.cascades[1])],
            render,
        }
    }

    fn create_resizables(target_size: (u32, u32), config: LightingQualityConfig) -> ResizeResults {
        let device = crate::Renderer::device();

        let light_tex_size = wgpu::Extent3d {
            width: target_size.0,
            height: target_size.1,
            depth_or_array_layers: 1,
        };

        // compute needed probe and cascade count to get full texture dimensions

        let screen_diag = uv::Vec2::new(target_size.0 as f32, target_size.1 as f32).mag();
        // iterative computation for cascade count,
        // the number needed is the minimum where a probe reaches all the way across the screen
        let range_c0 = config.probe_range();
        let mut range = range_c0;
        let mut cascade_count = 1;
        while range < screen_diag {
            // each probe's range starts at the previous one's
            // and has a length of 4 times the previous
            range += range_c0 * (4u32.pow(cascade_count)) as f32;
            cascade_count += 1;
        }

        let light_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("lights"),
            dimension: wgpu::TextureDimension::D2,
            size: light_tex_size,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
            mip_level_count: cascade_count,
            sample_count: 1,
        });
        let light_mips = (0..cascade_count)
            .map(|mip_idx| {
                light_tex.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: mip_idx,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let light_full = light_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let probe_count = [
            (target_size.0 as f32 / config.probe_interval).floor() as u32,
            (target_size.1 as f32 / config.probe_interval).floor() as u32,
        ];

        let cascade_tex = || {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("radiance cascades"),
                dimension: wgpu::TextureDimension::D2,
                size: wgpu::Extent3d {
                    width: probe_count[0],
                    height: probe_count[1],
                    depth_or_array_layers: 1,
                },
                format: CASCADE_TEX_FMT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
                mip_level_count: 1,
                sample_count: 1,
            });
            tex.create_view(&wgpu::TextureViewDescriptor::default())
        };
        let cascades = [cascade_tex(), cascade_tex()];

        let cascade_params = (0..cascade_count)
            .map(|level| CascadeParams::for_level(level, cascade_count, probe_count, config))
            .collect();

        let render_params = RenderParams {
            probe_spacing: config.probe_interval,
            probe_range: range_c0,
            probe_count,
            skip_raymarch: config.skip_final_cascade as u32,
            _pad: 0,
        };

        ResizeResults {
            light_tex_size: (light_tex_size.width, light_tex_size.height),
            textures: Textures {
                light_full,
                light_mips,
                cascades,
            },
            cascade_params,
            render_params,
        }
    }

    /// Recompute needed cascade and probe counts when window size changes.
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        let device = crate::Renderer::device();
        let queue = crate::Renderer::queue();

        let res = Self::create_resizables(new_size.into(), self.quality_conf);
        // make room in the params buffer if we need more cascades than before
        if res.cascade_params.len() > self.cascade_count {
            self.buffers.cascade_params =
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("cascade compute params"),
                    contents: res.cascade_params.as_bytes(),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
        } else {
            let buf = &self.buffers.cascade_params;
            queue.write_buffer(buf, 0, res.cascade_params.as_bytes());
        }
        self.cascade_count = res.cascade_params.len();
        self.light_tex_size = res.light_tex_size;
        queue.write_buffer(&self.buffers.render_params, 0, res.render_params.as_bytes());

        self.probe_count = res.cascade_params[0].probe_count;
        self.bind_groups = Self::create_bind_groups(
            self.cascade_count,
            &self.bind_group_layouts,
            &res.textures,
            &self.buffers,
            &self.bilinear_samp,
        );
        self.textures = res.textures;

        self.last_screen_size = new_size;
    }

    pub fn set_quality(&mut self, conf: LightingQualityConfig) {
        let needs_resize = self.quality_conf != conf;
        self.quality_conf = conf;
        if needs_resize {
            self.resize(self.last_screen_size);
        }
    }

    pub fn compute_light_mips<'pass>(
        &'pass self,
        pass: &mut wp::OwningScope<'_, wgpu::ComputePass<'pass>>,
    ) {
        let device = crate::Renderer::device();
        let mut pass = pass.scope("light mip chain", device);

        pass.set_pipeline(&self.pipelines.light_mip);
        // mip level count is equal to cascade count
        for mip_idx in 0..self.cascade_count as u32 - 1 {
            let mut pass = pass.scope(format!("mip level {mip_idx}"), device);

            let tiles_x =
                (self.light_tex_size.0 as f32 / ((mip_idx + 1) * TILE_SIZE) as f32).ceil() as u32;
            let tiles_y =
                (self.light_tex_size.1 as f32 / ((mip_idx + 1) * TILE_SIZE) as f32).ceil() as u32;

            pass.set_bind_group(0, &self.bind_groups.light_mips[mip_idx as usize], &[]);
            pass.dispatch_workgroups(tiles_x, tiles_y, 1);
        }
    }

    pub fn compute_gi(
        &self,
        scope: &mut wp::Scope<'_, wgpu::CommandEncoder>,
        camera: &crate::Camera,
    ) {
        let device = crate::Renderer::device();
        let queue = crate::Renderer::queue();

        let mut scope = scope.scope("radiance cascades", device);

        let frame_params = FrameParams {
            pixel_size_10cm: 10. / camera.pixels_per_world_unit(self.light_tex_size),
        };
        queue.write_buffer(&self.buffers.frame_params, 0, frame_params.as_bytes());

        // cascades starting with the last
        for casc_idx in (1..self.cascade_count).rev() {
            let mut pass = scope.scoped_render_pass(
                format!("cascade {casc_idx}"),
                device,
                wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.textures.cascades[(casc_idx + 1) % 2],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                },
            );

            pass.set_pipeline(&self.pipelines.cascade);
            pass.set_bind_group(0, &self.bind_groups.light_full, &[]);
            pass.set_bind_group(
                1,
                &self.bind_groups.cascades[casc_idx % 2],
                &[(casc_idx * size_of::<CascadeParams>()) as u32],
            );
            pass.draw(0..3, 0..1);
        }
    }
}
