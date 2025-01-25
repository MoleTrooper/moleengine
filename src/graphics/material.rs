use std::{mem::size_of, sync::OnceLock};

use wgpu::util::DeviceExt;
use zerocopy::{AsBytes, FromBytes};

static RESOURCES: OnceLock<MaterialResources> = OnceLock::new();
// white material that gets loaded if a mesh has no material
static DEFAULT_MATERIAL: OnceLock<Material> = OnceLock::new();

/// Singleton collection of shared resources.
struct MaterialResources {
    /// Bind group layout shared by all materials.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// White texture to bind when the material doesn't have a texture.
    pub blank_texture: Texture,
    /// Normal map facing directly in the normal direction,
    /// to bind when the material doesn't have a normal map.
    pub blank_normal: Texture,
}

impl MaterialResources {
    fn get<'a>() -> &'a Self {
        RESOURCES.get_or_init(Self::new)
    }

    fn new() -> Self {
        let device = crate::Renderer::device();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &[
                // parameter uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(size_of::<MaterialUniforms>() as _),
                    },
                    count: None,
                },
                // texture and sampler for diffuse
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // same for normal map
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let blank_texture = TextureData {
            label: Some("blank".to_string()),
            pixels: &[255, 255, 255, 255],
            format: wgpu::TextureFormat::Rgba8Unorm,
            dimensions: (1, 1),
        }
        .upload();

        let blank_normal = TextureData {
            label: Some("blank normal".to_string()),
            pixels: &[127, 127, 255, 0],
            format: wgpu::TextureFormat::Rgba8Unorm,
            dimensions: (1, 1),
        }
        .upload();

        Self {
            bind_group_layout,
            blank_texture,
            blank_normal,
        }
    }
}

/// Creation parameters for a material.
#[derive(Debug, Clone, Default)]
pub struct MaterialParams<'a> {
    /// An optional name, used to look the material up
    /// with [`GraphicsManager::get_material_id`][`super::GraphicsManager::get_material_id`].
    pub name: Option<&'a str>,
    /// A constant diffuse color.
    ///
    /// If this and `diffuse_tex` are both set,
    /// the texture values are multiplied by this value.
    pub base_color: Option<[f32; 4]>,
    /// Amount of light emitted by the material per unit of distance.
    pub emissive_color: Option<[f32; 4]>,
    /// Parameters for how the material absorbs light.
    pub attenuation: Option<AttenuationParams>,
    /// Texture data for the diffuse color.
    pub diffuse_tex: Option<TextureData<'a>>,
    /// Texture data for the normal map.
    pub normal_tex: Option<TextureData<'a>>,
}

/// Parameters controlling how a material absorbs light.
#[derive(Clone, Copy, Debug)]
pub struct AttenuationParams {
    /// Color that white light will become
    /// after moving through a `distance`-sized unit of the material.
    pub color: [f32; 3],
    /// Distance within the material that it takes
    /// for white light to turn into `self.color`.
    ///
    /// Due to limitations in the GPU representation of this information,
    /// this value is limited to the range 0..=1.
    /// This has not been found to be particularly limiting in practice.
    pub distance: f32,
}

impl Default for AttenuationParams {
    fn default() -> Self {
        Self {
            color: [1.; 3],
            distance: 1.,
        }
    }
}

/// A material determines the color and lighting properties of a mesh.
pub struct Material {
    pub(crate) participates_in_lighting: bool,
    pub(crate) bind_group: wgpu::BindGroup,
    // textures and buffer stored to avoid dropping them
    _uniform_buf: wgpu::Buffer,
    _diffuse_tex: Option<Texture>,
    _normal_tex: Option<Texture>,
}

impl Material {
    pub(super) fn new(params: MaterialParams) -> Self {
        let device = crate::Renderer::device();
        let res = MaterialResources::get();

        let diffuse_tex = params.diffuse_tex.map(|t| t.upload());
        let normal_tex = params.normal_tex.map(|t| t.upload());

        let diffuse = diffuse_tex.as_ref().unwrap_or(&res.blank_texture);
        let normal = normal_tex.as_ref().unwrap_or(&res.blank_normal);

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material uniforms"),
            contents: MaterialUniforms {
                base_color: params.base_color.unwrap_or([1.; 4]),
                emissive_color: params.emissive_color.unwrap_or([0.; 4]),
                attenuation_color: params.attenuation.unwrap_or_default().color,
                attenuation_distance: params.attenuation.unwrap_or_default().distance,
            }
            .as_bytes(),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &res.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(uniform_buf.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&diffuse.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&diffuse.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&normal.view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&normal.sampler),
                },
            ],
        });

        Self {
            participates_in_lighting: params.emissive_color.is_some()
                || params.attenuation.is_some(),
            bind_group,
            _uniform_buf: uniform_buf,
            _diffuse_tex: diffuse_tex,
            _normal_tex: normal_tex,
        }
    }

    pub(super) fn get_default<'a>() -> &'a Self {
        DEFAULT_MATERIAL.get_or_init(|| {
            Self::new(MaterialParams {
                name: None,
                base_color: Some([1.; 4]),
                emissive_color: None,
                attenuation: None,
                diffuse_tex: None,
                normal_tex: None,
            })
        })
    }

    #[inline]
    pub(crate) fn bind_group_layout<'a>() -> &'a wgpu::BindGroupLayout {
        &MaterialResources::get().bind_group_layout
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, AsBytes, FromBytes)]
struct MaterialUniforms {
    base_color: [f32; 4],
    emissive_color: [f32; 4],
    attenuation_color: [f32; 3],
    attenuation_distance: f32,
}

#[derive(Debug)]
pub struct Texture {
    pub(crate) _texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) sampler: wgpu::Sampler,
}

#[derive(Debug, Clone)]
pub struct TextureData<'a> {
    // this is a string due to complications in glTF loading that a &str would cause here
    pub label: Option<String>,
    pub pixels: &'a [u8],
    pub format: wgpu::TextureFormat,
    pub dimensions: (u32, u32),
}

impl TextureData<'_> {
    pub fn upload(self) -> Texture {
        let device = crate::Renderer::device();
        let queue = crate::Renderer::queue();
        let label = self.label.as_deref();
        let size = wgpu::Extent3d {
            width: self.dimensions.0,
            height: self.dimensions.1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            self.pixels,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // TODO: get the sampler from gltf
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Texture {
            _texture: texture,
            view,
            sampler,
        }
    }
}
