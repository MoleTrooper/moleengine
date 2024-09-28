use crate::math::{uv, Pose};

use itertools::izip;
use std::mem::size_of;
use zerocopy::{AsBytes, FromBytes};

use super::util::{GpuVec3, GpuVec4};

/// Primary light source covering the entire screen.
/// There can only be one active at a time.
///
/// A directional light has no position,
/// instead casting parallel rays over the entire scene.
/// This emulates a distant, powerful point light source like the sun.
#[derive(Clone, Copy, Debug)]
pub struct DirectionalLight {
    /// Color of the light. Default: white.
    pub color: [f32; 3],
    /// Direction in which the light rays travel. Default: positive z-axis.
    pub direction: uv::Vec3,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0],
            direction: uv::Vec3::new(0.0, 0.0, 1.0),
        }
    }
}

/// A point-like light source. There can be many in a scene.
///
/// Can be created on its own or attached to entities in a [`hecs::World`].
/// To get all lights in a world, use [`gather_from_world`][Self::gather_from_world].
///
/// The parameters `radius`, `falloff`, and `cutoff` control the shape of the light.
/// For an interactive visualization of the way these work,
/// see [this Desmos calculator](https://www.desmos.com/calculator/f7impirvum).
#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    /// Position of the light. Default: origin.
    ///
    /// If the light is attached to an entity
    /// and acquired with [`gather_from_world`][Self::gather_from_world],
    /// the position is interpreted to be relative to the entity's [`Pose`].
    /// Otherwise, this is in world space.
    pub position: uv::Vec3,
    /// Color of the light. Default: white.
    pub color: [f32; 3],
    /// Maximum radius the light can reach. Default: 10.
    ///
    /// At this distance, the light intensity will reach
    /// the cutoff value defined in `cutoff`.
    /// Light reaching further won't be rendered,
    /// so there will be a slight border visible at this distance
    /// unless `cutoff` is set very low.
    pub radius: f32,
    /// Falloff rate of light intensity. Default: 5.
    ///
    /// The higher this is, the faster the light intensity
    /// will decrease with distance.
    pub falloff: f32,
    /// Cutoff value for light intensity at a distance of `radius`.
    pub cutoff: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: uv::Vec3::zero(),
            color: [1.; 3],
            radius: 10.,
            falloff: 5.,
            cutoff: 1. / 256.,
        }
    }
}

impl PointLight {
    /// Get all point lights attached to entities in a world.
    ///
    /// If the entity has a [`Pose`],
    /// the position of the light is set to `pose * light.position`
    /// (i.e. the original light position is interpreted as being in the entity's local space),
    /// otherwise the light is returned as is.
    pub fn gather_from_world(world: &mut hecs::World) -> impl '_ + Iterator<Item = PointLight> {
        world
            .query_mut::<(&PointLight, Option<&Pose>)>()
            .into_iter()
            .map(|(_, (light, pose))| {
                let mut ret = *light;
                if let Some(&pose) = pose {
                    let pos_with_offset = pose * ret.position;
                    ret.position = pos_with_offset;
                }
                ret
            })
    }
}

/// Gpu-side representation of a directional light.
#[repr(C)]
#[derive(Debug, Clone, Copy, AsBytes, FromBytes)]
struct GpuDirectionalLight {
    color: GpuVec4,
    direction: GpuVec4,
}

impl From<DirectionalLight> for GpuDirectionalLight {
    fn from(l: DirectionalLight) -> Self {
        Self {
            color: l.color.into(),
            direction: l.direction.into(),
        }
    }
}

pub(crate) struct MainLights {
    pub ambient_color: [f32; 3],
    pub dir_lights: Vec<DirectionalLight>,
}

/// Gpu-side representation of a single point light.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, AsBytes, FromBytes)]
pub(crate) struct GpuPointLight {
    position: GpuVec4,
    color: GpuVec3,
    radius: f32,
    attn_linear: f32,
    attn_quadratic: f32,
    _pad: [u32; 2],
}

impl From<PointLight> for GpuPointLight {
    fn from(light: PointLight) -> Self {
        // see https://www.desmos.com/calculator/f7impirvum
        let attn_linear = light.falloff / light.radius;
        let intensity = light.color.into_iter().max_by(f32::total_cmp).unwrap();
        let attn_quadratic =
            (1. / light.radius.powi(2)) * ((intensity / light.cutoff) - light.cutoff - attn_linear);
        Self {
            position: light.position.into(),
            color: light.color.into(),
            radius: light.radius,
            attn_linear,
            attn_quadratic,
            _pad: [0; 2],
        }
    }
}

const MAX_POINT_LIGHTS: usize = 1024;
/// pixels per light bin
const TILE_SIZE: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, Debug, AsBytes, FromBytes)]
struct GpuPointLightBuffer {
    count: u32,
    tiles_x: u32,
    tiles_y: u32,
    _pad: u32,
    lights: [GpuPointLight; MAX_POINT_LIGHTS],
}

impl Default for GpuPointLightBuffer {
    fn default() -> Self {
        Self {
            count: 0,
            tiles_x: 0,
            tiles_y: 0,
            _pad: 0,
            // default is not derived for arrays this big so we need the manual impl
            lights: [GpuPointLight::default(); 1024],
        }
    }
}

/// GPU-side structures for lighting.
pub(crate) struct LightManager {
    main_light_buf: wgpu::Buffer,
    dir_light_count: usize,
    point_light_buf: wgpu::Buffer,
    light_bin_buf: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    cull_pipeline: wgpu::ComputePipeline,
    tile_count: (u32, u32),
}

impl LightManager {
    pub fn new() -> Self {
        let device = crate::Renderer::device();
        // manual size computation needed due to dynamically sized array
        let min_main_light_buf_size = 4 * 4 + 4 + size_of::<GpuDirectionalLight>() as u64;
        let main_light_buf = device.create_buffer(&wgpu::BufferDescriptor {
            size: min_main_light_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            label: Some("main lights"),
            mapped_at_creation: false,
        });

        let point_buf_size = size_of::<GpuPointLightBuffer>() as u64;
        let point_light_buf = device.create_buffer(&wgpu::BufferDescriptor {
            size: point_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            label: Some("point lights"),
            mapped_at_creation: false,
        });

        let tile_count = Self::tile_count_for_target(crate::Renderer::window().inner_size().into());
        let light_bin_buf = Self::create_light_bin_buf(tile_count);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // point lights
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(point_buf_size),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // point light index bins
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(4 * MAX_POINT_LIGHTS as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // directional light
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(min_main_light_buf_size),
                    },
                    count: None,
                },
            ],
            label: Some("lights"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lights"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: point_light_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_bin_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: main_light_buf.as_entire_binding(),
                },
            ],
        });

        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("light culling"),
            bind_group_layouts: &[
                crate::Camera::bind_group_layout(),
                &bind_group_layout,
                crate::Renderer::depth_bind_group_layout(),
            ],
            push_constant_ranges: &[],
        });

        let shader =
            device.create_shader_module(wgpu::include_wgsl!("./shaders/light_culling.wgsl"));

        let cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("light culling"),
            module: &shader,
            entry_point: "main",
            layout: Some(&pl_layout),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Self {
            main_light_buf,
            dir_light_count: 1,
            point_light_buf,
            light_bin_buf,
            bind_group_layout,
            bind_group,
            cull_pipeline,
            tile_count,
        }
    }

    fn tile_count_for_target(target_dimensions: (u32, u32)) -> (u32, u32) {
        (
            (target_dimensions.0 / TILE_SIZE as u32) + 1,
            (target_dimensions.1 / TILE_SIZE as u32) + 1,
        )
    }

    /// Recreate the light culling buffer when window size changes.
    pub fn recreate_light_bins(&mut self, target_dimensions: (u32, u32)) {
        self.tile_count = Self::tile_count_for_target(target_dimensions);
        self.light_bin_buf = Self::create_light_bin_buf(self.tile_count);
        self.recreate_bind_group();
    }

    fn recreate_bind_group(&mut self) {
        let device = crate::Renderer::device();
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lights"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.point_light_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.light_bin_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.main_light_buf.as_entire_binding(),
                },
            ],
        });
    }

    fn create_light_bin_buf(tile_count: (u32, u32)) -> wgpu::Buffer {
        let device = crate::Renderer::device();

        device.create_buffer(&wgpu::BufferDescriptor {
            size: (tile_count.0 * tile_count.1) as u64 * MAX_POINT_LIGHTS as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            label: Some("point light bins"),
            mapped_at_creation: false,
        })
    }

    /// Write the directional and ambient light information to the GPU.
    pub fn write_main_lights(&mut self, main_lights: MainLights) {
        let queue = crate::Renderer::queue();

        // recreate the main light buffer
        // if the number of directional lights increases past capacity
        // (most of the time the number won't be changing at all so this will be rare)
        let dir_light_count = main_lights.dir_lights.len();
        let buf_size = 4 * 4 + 4 + (dir_light_count * size_of::<GpuDirectionalLight>());
        if dir_light_count > self.dir_light_count {
            let device = crate::Renderer::device();
            self.main_light_buf = device.create_buffer(&wgpu::BufferDescriptor {
                size: buf_size as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                label: Some("main lights"),
                mapped_at_creation: false,
            });
            self.dir_light_count = dir_light_count;
            self.recreate_bind_group();
        }

        // there's a dynamically-sized array here
        // so we can't directly convert a cpu-side type to bytes.
        // `encase` could do this for us
        // but I can't be bothered to pull that in for this one case,
        // so do it manually
        let mut bytes = Vec::with_capacity(buf_size);
        bytes.extend_from_slice(GpuVec3::from(main_lights.ambient_color).as_bytes());
        bytes.extend_from_slice((dir_light_count as u32).as_bytes());
        for dir_light in main_lights.dir_lights {
            bytes.extend_from_slice(GpuDirectionalLight::from(dir_light).as_bytes());
        }
        queue.write_buffer(&self.main_light_buf, 0, &bytes);
    }

    /// Write point lights to the GPU.
    pub fn write_point_lights(&self, mut lights: Vec<GpuPointLight>) {
        let queue = crate::Renderer::queue();

        lights.truncate(MAX_POINT_LIGHTS);

        let mut contents = GpuPointLightBuffer {
            count: lights.len() as u32,
            tiles_x: self.tile_count.0,
            tiles_y: self.tile_count.1,
            ..Default::default()
        };
        for (src, dst) in izip!(&lights[..], &mut contents.lights[..]) {
            *dst = *src;
        }

        queue.write_buffer(&self.point_light_buf, 0, contents.as_bytes());
    }

    /// Run the compute shader assigning lights to tiles on the screen.
    pub fn cull_lights<'pass>(
        &'pass self,
        pass: &mut wgpu::ComputePass<'pass>,
        camera: &'pass crate::Camera,
        depth_bind_group: &'pass wgpu::BindGroup,
    ) {
        pass.set_pipeline(&self.cull_pipeline);
        pass.set_bind_group(0, &camera.bind_group, &[]);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.set_bind_group(2, depth_bind_group, &[]);
        pass.dispatch_workgroups(self.tile_count.0, self.tile_count.1, 1);
    }
}
