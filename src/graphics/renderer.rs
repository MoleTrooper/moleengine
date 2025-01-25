use super::{
    gi,
    line_renderer::LineRenderer,
    mesh::{skin::SkinPipeline, MeshRenderer},
};
use std::sync::OnceLock;

use wgpu_profiler as wp;

// there is only ever one wgpu context,
// and since the device and queue are frequently needed to create resources,
// we store those globally here
// so that the user doesn't have to ferry them around constantly

static DEVICE: OnceLock<wgpu::Device> = OnceLock::new();
static QUEUE: OnceLock<wgpu::Queue> = OnceLock::new();
static WINDOW: OnceLock<winit::window::Window> = OnceLock::new();

pub const SWAPCHAIN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth16Unorm;
pub const MSAA_SAMPLES: u32 = 4;
pub const DEFAULT_MULTISAMPLE_STATE: wgpu::MultisampleState = wgpu::MultisampleState {
    count: MSAA_SAMPLES,
    mask: !0,
    alpha_to_coverage_enabled: false,
};

/// A Renderer manages resources needed to draw graphics to the screen.
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    window_scale_factor: f64,

    msaa_view: wgpu::TextureView,
    // textures and bind group for depth and lights to use in GI
    depth_tex: wgpu::Texture,
    depth_view: wgpu::TextureView,
    emissive_tex: wgpu::Texture,
    emissive_view: wgpu::TextureView,

    gi_pipeline: gi::GlobalIlluminationPipeline,
    mesh_renderer: MeshRenderer,
    skin_pl: SkinPipeline,
    // rendering subsystems that aren't always used in lazily initialized Options
    // so we can have a unified API to call them through `Frame`
    // but don't pay for them if the user doesn't use them
    line_renderer: Option<LineRenderer>,

    pub(crate) profiler: wp::GpuProfiler,
}

/// An error that occurred during renderer initialization.
#[derive(thiserror::Error, Debug)]
pub enum RendererInitError {
    #[error("Failed to get a window handle")]
    HandleError,
    #[error("Failed to create surface")]
    CreateSurfaceError(#[from] wgpu::CreateSurfaceError),
    #[error("Adapter request failed")]
    RequestAdapterError,
    #[error("Device request failed")]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
    #[error("Another Renderer already existed")]
    AlreadyInitialized,
    #[error("Failed to create a profiler")]
    ProfilerError(#[from] wp::CreationError),
}

impl Renderer {
    /// Create a Renderer.
    /// The [`Game`][crate::game::Game] API does this automatically.
    pub(crate) async fn init(
        window: winit::window::Window,
        config: crate::GraphicsConfig,
    ) -> Result<Self, RendererInitError> {
        let instance = wgpu::Instance::default();
        let surface = unsafe {
            instance.create_surface_unsafe(
                wgpu::SurfaceTargetUnsafe::from_window(&window)
                    .map_err(|_| RendererInitError::HandleError)?,
            )?
        };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .ok_or(RendererInitError::RequestAdapterError)?;

        #[cfg(feature = "tracy")]
        let profiling_features = wp::GpuProfiler::ALL_WGPU_TIMER_FEATURES;
        #[cfg(not(feature = "tracy"))]
        let profiling_features = wgpu::Features::empty();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::default() | profiling_features,
                    required_limits: wgpu::Limits {
                        min_uniform_buffer_offset_alignment: 64,
                        ..Default::default()
                    },
                    label: None,
                },
                None,
            )
            .await?;

        let window_size = window.inner_size();

        let swapchain_capabilities = surface.get_capabilities(&adapter);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: SWAPCHAIN_FORMAT,
            width: window_size.width,
            height: window_size.height,
            present_mode: if config.use_vsync {
                wgpu::PresentMode::AutoVsync
            } else {
                wgpu::PresentMode::AutoNoVsync
            },
            alpha_mode: swapchain_capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let window_scale_factor = window.scale_factor();

        #[cfg(feature = "tracy")]
        let profiler = wp::GpuProfiler::new_with_tracy_client(
            wp::GpuProfilerSettings::default(),
            adapter.get_info().backend,
            &device,
            &queue,
        )?;
        // if tracy is not enabled, create a no-op profiler anyway
        // so that we don't need to feature-gate all profiler usages
        #[cfg(not(feature = "tracy"))]
        let profiler = wp::GpuProfiler::new(wp::GpuProfilerSettings {
            enable_timer_queries: false,
            enable_debug_groups: false,
            ..Default::default()
        })?;

        DEVICE
            .set(device)
            .map_err(|_| RendererInitError::AlreadyInitialized)?;
        QUEUE
            .set(queue)
            .map_err(|_| RendererInitError::AlreadyInitialized)?;
        WINDOW
            .set(window)
            .map_err(|_| RendererInitError::AlreadyInitialized)?;

        let msaa_tex = Self::create_msaa_texture(window_size);
        let msaa_view = msaa_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_tex = Self::create_depth_texture(window_size);
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let emissive_tex = Self::create_emissive_texture(window_size);
        let emissive_view = emissive_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let gi_pipeline = gi::GlobalIlluminationPipeline::new(config.lighting_quality);
        let mesh_renderer = MeshRenderer::new(&gi_pipeline);
        let skin_pl = SkinPipeline::new();

        Ok(Renderer {
            surface,
            surface_config,
            window_scale_factor,
            msaa_view,
            depth_tex,
            depth_view,
            emissive_tex,
            emissive_view,
            gi_pipeline,
            mesh_renderer,
            skin_pl,
            line_renderer: None,
            profiler,
        })
    }

    /// Get a reference to the the window the game draws to.
    /// # Panics
    /// This function panics if the renderer hasn't been initialized yet,
    /// i.e. if [`Game::run`][crate::Game::run] hasn't been called yet.
    pub fn window<'a>() -> &'a winit::window::Window {
        WINDOW.get().expect("Renderer has not been initialized yet")
    }

    /// Get a reference to the the global device instance.
    /// # Panics
    /// This function panics if the renderer hasn't been initialized yet,
    /// i.e. if [`Game::run`][crate::Game::run] hasn't been called yet.
    #[inline]
    pub fn device<'a>() -> &'a wgpu::Device {
        DEVICE.get().expect("Renderer has not been initialized yet")
    }

    /// Get a reference to the the global queue instance.
    /// # Panics
    /// This function panics if the renderer hasn't been initialized yet,
    /// i.e. if [`Game::run`][crate::Game::run] hasn't been called yet.
    #[inline]
    pub fn queue<'a>() -> &'a wgpu::Queue {
        QUEUE.get().expect("Renderer has not been initialized yet")
    }

    /// Change the size of the frame `draw_to_window` draws into.
    /// This is called automatically by the gameloop when the window size changes.
    pub(crate) fn resize_swap_chain(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size == self.window_size() {
            return;
        }
        let device = Self::device();
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(device, &self.surface_config);
        let msaa_tex = Self::create_msaa_texture(new_size);
        self.msaa_view = msaa_tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.depth_tex = Self::create_depth_texture(new_size);
        self.depth_view = self
            .depth_tex
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.emissive_tex = Self::create_emissive_texture(new_size);
        self.emissive_view = self
            .emissive_tex
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.gi_pipeline.resize(new_size);
    }

    fn create_msaa_texture(size: winit::dpi::PhysicalSize<u32>) -> wgpu::Texture {
        let device = Self::device();
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("msaa"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: SWAPCHAIN_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    fn create_depth_texture(size: winit::dpi::PhysicalSize<u32>) -> wgpu::Texture {
        let device = Self::device();
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    fn create_emissive_texture(size: winit::dpi::PhysicalSize<u32>) -> wgpu::Texture {
        let device = Self::device();
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("emissive"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    #[inline]
    pub fn swapchain_format(&self) -> wgpu::TextureFormat {
        SWAPCHAIN_FORMAT
    }

    #[inline]
    pub fn depth_format(&self) -> wgpu::TextureFormat {
        DEPTH_FORMAT
    }

    /// Get the size of the window this Renderer draws to in pixels.
    #[inline]
    pub fn window_size(&self) -> winit::dpi::PhysicalSize<u32> {
        winit::dpi::PhysicalSize::new(self.surface_config.width, self.surface_config.height)
    }

    /// Get the scale factor of the window this Renderer draws to.
    #[inline]
    pub fn window_scale_factor(&self) -> f64 {
        self.window_scale_factor
    }

    /// Depth-stencil state that uses the same depth format as the window depth buffer
    /// and writes depths to the buffer.
    #[inline]
    pub fn default_depth_stencil_state(&self) -> wgpu::DepthStencilState {
        wgpu::DepthStencilState {
            format: self.depth_tex.format(),
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }
    }

    /// Change lighting quality parameters at runtime.
    #[inline]
    pub fn set_lighting_quality(&mut self, conf: gi::LightingQualityConfig) {
        self.gi_pipeline.set_quality(conf);
    }

    /// Set the environment map for lighting.
    #[inline]
    pub fn set_environment_map(&mut self, params: &crate::EnvironmentMap) {
        self.gi_pipeline.env_map.bake(params);
    }

    /// Start drawing a frame.
    #[inline]
    pub fn begin_frame(&mut self) -> Frame<'_> {
        let surface = self
            .surface
            .get_current_texture()
            .expect("Failed to get next swap chain texture");
        let view = surface
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let device = Self::device();
        let encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        Frame {
            renderer: self,
            encoder: Some(encoder),
            surface: Some(surface),
            target_view: view,
            clear_color: Some(wgpu::Color::BLACK),
        }
    }
}

pub struct Frame<'a> {
    renderer: &'a mut Renderer,
    // encoder and surface in Options
    // so that we can take them out on drop without using unsafe
    encoder: Option<wgpu::CommandEncoder>,
    surface: Option<wgpu::SurfaceTexture>,
    target_view: wgpu::TextureView,
    clear_color: Option<wgpu::Color>,
}

impl Frame<'_> {
    /// Set the color the framebuffer will be cleared with
    /// when the shading is executed. Black by default.
    pub fn set_clear_color(&mut self, color: [f32; 4]) {
        self.clear_color = Some(wgpu::Color {
            r: color[0] as f64,
            g: color[1] as f64,
            b: color[2] as f64,
            a: color[3] as f64,
        });
    }

    /// Draw all meshes in the world.
    pub fn draw_meshes(
        &mut self,
        manager: &mut crate::GraphicsManager,
        world: &mut hecs::World,
        camera: &crate::Camera,
    ) {
        let device = Renderer::device();
        let encoder = self.encoder.as_mut().unwrap();
        let mut scope = self.renderer.profiler.scope("draw meshes", encoder, device);

        // compute skins

        {
            let mut cpass = scope.scoped_compute_pass("compute skins", device);
            self.renderer.skin_pl.compute_skins(&mut cpass, manager);
        }

        // upload mesh data

        self.renderer.mesh_renderer.prepare(manager, world);

        // render depth

        {
            let mut rpass = scope.scoped_render_pass(
                "render depth",
                device,
                wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.renderer.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                },
            );

            let mesh_rend = &mut self.renderer.mesh_renderer;
            mesh_rend.depth_pass(&mut rpass, manager, camera);
        }

        // render light emitters and occluders

        {
            let mut rpass = scope.scoped_render_pass(
                "render lights",
                device,
                wgpu::RenderPassDescriptor {
                    label: Some("lights"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.renderer.gi_pipeline.textures.light_emission,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.renderer.gi_pipeline.textures.light_attenuation,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                // clear with a color that corresponds to a fully transparent material
                                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                },
            );

            let mesh_rend = &mut self.renderer.mesh_renderer;
            mesh_rend.emissive_pass(&mut rpass, manager, camera);
        }

        // compute global illumination

        {
            let mut cpass = scope.scoped_compute_pass("compute light mips", device);
            self.renderer.gi_pipeline.compute_light_mips(&mut cpass);
        }

        self.renderer.gi_pipeline.compute_gi(&mut scope, camera);

        // final render

        {
            let mut rpass = scope.scoped_render_pass(
                "render meshes",
                device,
                wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.renderer.msaa_view,
                        resolve_target: Some(&self.target_view),
                        ops: Self::ops(self.clear_color.take()),
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.renderer.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                },
            );

            let mesh_rend = &mut self.renderer.mesh_renderer;
            mesh_rend.draw_pass(&mut rpass, manager, camera, &self.renderer.gi_pipeline);
        }
    }

    /// Draw a collection of line strips with the line renderer.
    pub fn draw_lines<'s>(
        &mut self,
        manager: &crate::GraphicsManager,
        camera: &crate::Camera,
        lines: impl IntoIterator<Item = &'s super::line_renderer::LineStrip>,
    ) {
        let device = Renderer::device();
        let encoder = self.encoder.as_mut().unwrap();
        let mut scope = self.renderer.profiler.scope("draw lines", encoder, device);

        let line_rend = self
            .renderer
            .line_renderer
            .get_or_insert_with(LineRenderer::new);

        let mut pass = scope.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lines"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.renderer.msaa_view,
                resolve_target: Some(&self.target_view),
                ops: Self::ops(self.clear_color.take()),
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.renderer.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        for line in lines {
            line_rend.draw(&mut pass, manager, camera, line);
        }
    }

    /// Begin a render pass with default parameters that draws to the screen.
    pub fn pass(&mut self) -> wgpu::RenderPass<'_> {
        let encoder = self.encoder.as_mut().unwrap();
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.renderer.msaa_view,
                resolve_target: Some(&self.target_view),
                ops: Self::ops(self.clear_color.take()),
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.renderer.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        })
    }

    fn ops(clear_color: Option<wgpu::Color>) -> wgpu::Operations<wgpu::Color> {
        wgpu::Operations {
            load: match clear_color {
                Some(color) => wgpu::LoadOp::Clear(color),
                None => wgpu::LoadOp::Load,
            },
            store: wgpu::StoreOp::Store,
        }
    }

    /// Access the command encoder recording commands for this frame.
    #[inline]
    pub fn encoder_mut(&mut self) -> &mut wgpu::CommandEncoder {
        self.encoder.as_mut().unwrap()
    }
}

impl Drop for Frame<'_> {
    fn drop(&mut self) {
        let queue = Renderer::queue();
        let mut encoder = self.encoder.take().unwrap();
        self.renderer.profiler.resolve_queries(&mut encoder);
        queue.submit(Some(encoder.finish()));
        self.surface.take().unwrap().present();
    }
}
