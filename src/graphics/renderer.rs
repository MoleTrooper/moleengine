/// A Renderer manages resources needed to draw graphics to the screen.
pub struct Renderer {
    pub device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface,
    surface_config: wgpu::SurfaceConfiguration,
    swapchain_format: wgpu::TextureFormat,
    window_scale_factor: f64,

    /// Depth buffer automatically kept in sync with the swapchain size.
    pub window_depth_buffer: super::DepthBuffer,

    // current active frame stored here instead of in RenderContext
    // so that we can interleave drawing to window and drawing to textures
    active_frame: Option<Frame>,
}

struct Frame {
    surface: wgpu::SurfaceTexture,
    view: wgpu::TextureView,
}

impl Renderer {
    /// Create a Renderer.
    /// The [`Game`][crate::game::Game] API does this automatically.
    pub(crate) async fn init(window: &winit::window::Window) -> Self {
        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Renderer init failed: failed to create adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await
            .expect("Failed to create wgpu device");

        let window_size = window.inner_size();

        // surface.get_preferred_format gives a non-SRGB format on wasm
        // which screws up colors. not sure if setting it to a constant
        // is the correct solution but it works on my machines :v)
        let swapchain_format = wgpu::TextureFormat::Bgra8UnormSrgb;

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: window_size.width,
            height: window_size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
        };
        surface.configure(&device, &surface_config);

        let depth_buffer =
            super::DepthBuffer::new(&device, window_size.into(), Some("global depth buffer"));

        Renderer {
            device,
            queue,
            surface,
            surface_config,
            swapchain_format,
            window_scale_factor: window.scale_factor(),
            window_depth_buffer: depth_buffer,
            active_frame: None,
        }
    }

    #[inline]
    pub fn swapchain_format(&self) -> wgpu::TextureFormat {
        self.swapchain_format
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

    /// Change the size of the frame `draw_to_window` draws into.
    /// This is called automatically by the gameloop when the window size changes.
    #[inline]
    pub(crate) fn resize_swap_chain(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
        self.window_depth_buffer =
            super::DepthBuffer::new(&self.device, new_size.into(), Some("global depth buffer"));
    }

    /// Begin drawing directly into the game window.
    pub fn draw_to_window(&mut self) -> RenderContext<'_> {
        // start a new frame if this is the first time we're drawing to the window
        // since last present
        if self.active_frame.is_none() {
            let surface = self
                .surface
                .get_current_texture()
                .expect("Failed to get next swap chain texture");
            let view = surface
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            self.active_frame = Some(Frame { surface, view });
        }
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let target_size = self.window_size().into();
        let queue = &mut self.queue;

        RenderContext {
            // active frame was just set so unwrap is safe
            target: RenderTarget {
                view: &self.active_frame.as_ref().unwrap().view,
                depth: Some(&self.window_depth_buffer.view),
            },
            encoder: CommandEncoder(encoder),
            device: &self.device,
            queue,
            target_size,
            submit_check: SubmitCheck::new(),
        }
    }

    /// Begin drawing to a non-screen texture, optionally with a self-provided depth texture.
    ///
    /// If you need the depth texture from the window, use
    /// [`draw_to_texture_window_depth`][Self::draw_to_texture_window_depth]
    pub fn draw_to_texture<'s, 'v: 's>(
        &'s mut self,
        view: &'v wgpu::TextureView,
        depth_target: Option<&'v wgpu::TextureView>,
        target_size: (u32, u32),
    ) -> RenderContext<'s> {
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let queue = &mut self.queue;

        RenderContext {
            target: RenderTarget {
                view,
                depth: depth_target,
            },
            encoder: CommandEncoder(encoder),
            device: &self.device,
            queue,
            target_size,
            submit_check: SubmitCheck::new(),
        }
    }

    /// Begin drawing to a non-screen texture, also using the depth buffer of the render window.
    pub fn draw_to_texture_window_depth<'s, 'v: 's>(
        &'s mut self,
        view: &'v wgpu::TextureView,
        target_size: (u32, u32),
    ) -> RenderContext<'s> {
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let queue = &mut self.queue;

        RenderContext {
            target: RenderTarget {
                view,
                depth: Some(&self.window_depth_buffer.view),
            },
            encoder: CommandEncoder(encoder),
            device: &self.device,
            queue,
            target_size,
            submit_check: SubmitCheck::new(),
        }
    }

    /// Display everything drawn to the window since the last `present_frame` call.
    /// Must be called at the end of every frame.
    pub fn present_frame(&mut self) {
        if let Some(frame) = self.active_frame.take() {
            frame.surface.present();
        }
    }
}

pub struct RenderTarget<'a> {
    pub view: &'a wgpu::TextureView,
    pub depth: Option<&'a wgpu::TextureView>,
}

/// An interface that lets you send draw instructions to the GPU.
///
/// You **must** call [`submit`](Self::submit) when you drop the context.
/// Not doing so would result in a memory leak, so
/// `RenderContext` will panic on drop if you do this.
///
/// TODOC: example
pub struct RenderContext<'a> {
    pub target: RenderTarget<'a>,
    pub encoder: CommandEncoder,
    pub device: &'a wgpu::Device,
    pub queue: &'a mut wgpu::Queue,
    pub target_size: (u32, u32),
    // this is just used to warn if a context was dropped without submitting.
    // doing that leaks memory
    submit_check: SubmitCheck,
}

impl<'a> RenderContext<'a> {
    /// Fill the render target with a flat color.
    ///
    /// If you need access to other fields of the RenderContext, this method also exists on the
    /// `encoder` so you can partial borrow when needed.
    #[inline]
    pub fn clear(&mut self, color: wgpu::Color) {
        self.encoder.clear(&self.target, color)
    }

    /// Begin a render pass that draws on top of what's already in the target
    /// and uses the depth buffer.
    ///
    /// If you need access to other fields of the RenderContext, this method also exists on the
    /// `encoder` so you can partial borrow when needed.
    #[inline]
    pub fn pass(&mut self, label: Option<&'static str>) -> wgpu::RenderPass {
        self.encoder.pass(&self.target, label)
    }

    /// Begin a render pass that draws on top of what's already in the target
    /// and ignores (i.e. doesn't bind at all) the depth buffer.
    ///
    /// If you need access to other fields of the RenderContext, this method also exists on the
    /// `encoder` so you can partial borrow when needed.
    #[inline]
    pub fn pass_without_depth(&mut self, label: Option<&'static str>) -> wgpu::RenderPass {
        self.encoder.pass_without_depth(&self.target, label)
    }

    /// Submit the commands made through this context to the GPU.
    /// Must be called or nothing is actually executed!
    pub fn submit(mut self) {
        self.queue.submit(Some(self.encoder.0.finish()));
        self.submit_check.0 = true;
    }
}

/// A wrapper around [`wgpu::CommandEncoder`][wgpu::CommandEncoder]
/// to facilitate creation of render passes with default parameters
/// while also partial borrowing other fields from [`RenderContext`][self::RenderContext].
pub struct CommandEncoder(pub wgpu::CommandEncoder);

impl CommandEncoder {
    /// Fill the render target with a flat color.
    pub fn clear(&mut self, target: &RenderTarget, color: wgpu::Color) {
        self.0.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(color),
                    store: true,
                },
            })],
            depth_stencil_attachment: target.depth.map(|depth| {
                wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: true,
                    }),
                }
            }),
        });
        // drop the pass immediately, causing the clear instruction
        // to be written to the encoder
    }

    /// Begin a render pass that draws on top of what's already in the target
    /// and uses the depth buffer.
    #[inline]
    pub fn pass<'s, 't: 's>(
        &'s mut self,
        target: &'s RenderTarget<'t>,
        label: Option<&'static str>,
    ) -> wgpu::RenderPass {
        self._pass(target, true, label)
    }

    /// Begin a render pass that draws on top of what's already in the target
    /// and ignores (i.e. doesn't bind at all) the depth buffer.
    #[inline]
    pub fn pass_without_depth<'s, 't: 's>(
        &'s mut self,
        target: &'s RenderTarget<'t>,
        label: Option<&'static str>,
    ) -> wgpu::RenderPass {
        self._pass(target, false, label)
    }

    fn _pass<'s, 't: 's>(
        &'s mut self,
        target: &'s RenderTarget<'t>,
        use_depth: bool,
        label: Option<&'static str>,
    ) -> wgpu::RenderPass {
        self.0.begin_render_pass(&wgpu::RenderPassDescriptor {
            label,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            })],
            depth_stencil_attachment: if !use_depth {
                None
            } else {
                target
                    .depth
                    .map(|depth| wgpu::RenderPassDepthStencilAttachment {
                        view: depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: true,
                        }),
                        stencil_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: true,
                        }),
                    })
            },
        })
    }
}

/// `RenderContext::submit` requires taking ownership and destructuring,
/// which makes submitting on drop too annoying.
/// Instead, use this to panic if the user drops a context wrong.
struct SubmitCheck(bool);
impl SubmitCheck {
    fn new() -> Self {
        Self(false)
    }
}
impl Drop for SubmitCheck {
    fn drop(&mut self) {
        assert!(self.0, "Dropped a RenderContext without submitting");
    }
}
