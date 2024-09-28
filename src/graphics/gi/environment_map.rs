use std::{f32::consts::TAU, mem::size_of};
use zerocopy::{AsBytes, FromBytes};

use crate::{
    graphics::util::{GpuVec2, GpuVec4},
    math::uv,
};

/// Width of the environment map texture.
const MAP_RESOLUTION: u32 = 256;

/// Maximum number of directional lights.
///
/// Usually there will be only one or two,
/// but allocate space for a handful
/// so we don't have to do dynamic buffer resizes.
///
/// (These are only used directly in the far-field shader
/// that doesn't use the GI lighting;
/// more lights can technically exist in GI)
const MAX_LIGHTS: usize = 10;

/// Parameters for additional lighting from off-screen sources
/// using a very simple gradient sky model.
/// Apply with [Renderer::set_environment_map][crate::Renderer::set_environment_map].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct EnvironmentMap {
    /// A constant ambient light applied everywhere in the scene.
    pub ambient: [f32; 3],
    /// Color of the horizon.
    pub horizon: [f32; 3],
    /// Color of the sky directly above.
    /// Colors for other angles are interpolated between this and `horizon`.
    pub zenith: [f32; 3],
    /// Color of the ground.
    pub ground: [f32; 3],
    /// Directional light sources (sun or moon, usually).
    pub lights: Vec<DirectionalLight>,
}

impl EnvironmentMap {
    /// Example set of values for a moonlit night scenario.
    pub fn preset_night() -> Self {
        Self {
            ambient: [0.005, 0.007, 0.008],
            horizon: [0.005, 0.011, 0.012],
            zenith: [0.006, 0.011, 0.017],
            ground: [0.010, 0.007, 0.006],
            lights: vec![DirectionalLight {
                direction: uv::Vec2::new(-0.2, -1.).normalized(),
                color: [0.075, 0.089, 0.090],
            }],
        }
    }

    /// Example set of values for a daytime scenario.
    pub fn preset_day() -> Self {
        Self {
            ambient: [0.008, 0.012, 0.014],
            horizon: [0.096, 0.297, 0.331],
            zenith: [0.032, 0.125, 0.236],
            ground: [0.026, 0.017, 0.014],
            lights: vec![DirectionalLight {
                direction: uv::Vec2::new(0.2, -1.).normalized(),
                color: [0.876, 0.829, 0.705],
            }],
        }
    }

    /// Example set of values for a sunset scenario.
    pub fn preset_sunset() -> Self {
        Self {
            ambient: [0.009, 0.013, 0.015],
            horizon: [0.206, 0.231, 0.129],
            zenith: [0.077, 0.264, 0.272],
            ground: [0.023, 0.013, 0.009],
            lights: vec![DirectionalLight {
                direction: uv::Vec2::new(0.7, -0.35).normalized(),
                color: [0.798, 0.370, 0.063],
            }],
        }
    }
}

/// Map definining additional lighting from off-screen sources.
pub struct EnvironmentMapData {
    // buffer containing the ambient color and directional lights for shading
    // (directional lights are used directly for background objects
    // outside of the GI field of influence,
    // otherwise lights are baked into the environment texture
    // and included during GI computation)
    pub(super) render_buf: wgpu::Buffer,
    map_tex: wgpu::Texture,
    pub(super) map_view: wgpu::TextureView,
    // parameters stored to check if we need to re-bake
    prev_params: EnvironmentMap,
}

/// Primary light source covering the entire screen.
///
/// A directional light has no position,
/// instead casting parallel rays over the entire scene.
/// This emulates a distant, powerful point light source like the sun.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionalLight {
    /// Color of the light. Default: white.
    pub color: [f32; 3],
    /// Direction in which the light rays travel. Default: negative y-axis.
    pub direction: uv::Vec2,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0],
            direction: -uv::Vec2::unit_y(),
        }
    }
}

/// Gpu-side representation of a directional light.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, AsBytes, FromBytes)]
struct GpuDirectionalLight {
    color: GpuVec4,
    direction: GpuVec2,
}

impl From<DirectionalLight> for GpuDirectionalLight {
    fn from(l: DirectionalLight) -> Self {
        Self {
            color: l.color.into(),
            direction: l.direction.normalized().into(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, AsBytes, FromBytes)]
pub(super) struct RenderData {
    ambient_color: GpuVec4,
    light_count: u32,
    lights: [GpuDirectionalLight; MAX_LIGHTS],
}

impl Default for EnvironmentMapData {
    fn default() -> Self {
        let device = crate::Renderer::device();
        let render_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("environment map render data"),
            size: size_of::<RenderData>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let map_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("environment map"),
            dimension: wgpu::TextureDimension::D1,
            size: wgpu::Extent3d {
                width: MAP_RESOLUTION,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let map_view = map_tex.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            render_buf,
            map_tex,
            map_view,
            prev_params: Default::default(),
        }
    }
}

impl EnvironmentMapData {
    pub fn bake(&mut self, params: &EnvironmentMap) {
        if *params == self.prev_params {
            return;
        }
        self.prev_params = params.clone();

        let queue = crate::Renderer::queue();

        // upload render data

        let render_data = RenderData {
            ambient_color: params.ambient.into(),
            light_count: params.lights.len() as u32,
            lights: std::array::from_fn(|i| {
                if i < params.lights.len() {
                    params.lights[i].into()
                } else {
                    Default::default()
                }
            }),
        };
        queue.write_buffer(&self.render_buf, 0, render_data.as_bytes());

        // bake environment map texture

        let tex_data = Self::generate_texture(params);
        queue.write_texture(
            self.map_tex.as_image_copy(),
            tex_data.as_bytes(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: None,
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: MAP_RESOLUTION,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    fn generate_texture(params: &EnvironmentMap) -> [[u8; 4]; MAP_RESOLUTION as usize] {
        // temporarily turn colors into vectors so we can easily do math on them
        // (TODO: use `palette` or a similar color library instead)
        let zenith = uv::Vec3::from(params.zenith);
        let horizon = uv::Vec3::from(params.horizon);
        let ground = uv::Vec3::from(params.ground);

        let angle_incr = TAU / MAP_RESOLUTION as f32;

        std::array::from_fn(|i| {
            let mut color = uv::Vec3::zero();

            // rotate clockwise to match the orientation of texture space
            let angle = TAU - i as f32 * angle_incr;
            let (sin, cos) = angle.sin_cos();
            let dir = uv::Vec2::new(cos, sin);
            if sin >= 0. {
                let sky_color = sin.abs() * zenith + (1. - sin.abs()) * horizon;
                color += sky_color;
            } else {
                color += ground;
            }

            for light in &params.lights {
                let dot_dir = -light.direction.normalized().dot(dir);
                if dot_dir <= 0. {
                    continue;
                }
                let light_col = uv::Vec3::from(light.color);
                // demoscene-inspired power formula
                // to approximate a sun-like circle light source in the distance.
                // the choice of power is arbitrary,
                // just made to "look like there's a light of the color I chose"
                // (higher power -> less observed intensity
                // due to fewer rays falling in the angle range)
                color += light_col * dot_dir.powi(5);
            }

            let as_u8 = |channel: f32| (u8::MAX as f32 * channel).round() as u8;
            [as_u8(color.x), as_u8(color.y), as_u8(color.z), 255]
        })
    }
}
