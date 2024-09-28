//! Tools for creating a windw and starting a managed game loop.

use std::collections::VecDeque;

use instant::Instant;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::{
    graphics::renderer::RendererInitError,
    physics::{hecs_sync::HecsSyncManager, ForceField, PhysicsWorld},
};

// time snapping technique from Tyler Glaiel's blog post
// https://medium.com/@tglaiel/how-to-make-your-game-run-at-60fps-24c61210fe75
const NANOS_120FPS: u128 = 1_000_000_000 / 120;
const NANOS_60FPS: u128 = 1_000_000_000 / 60;
const NANOS_30FPS: u128 = 1_000_000_000 / 30;
const NANOS_20FPS: u128 = 1_000_000_000 / 20;
const NANOS_15FPS: u128 = 1_000_000_000 / 15;
const SNAP_THRESHOLD: u128 = 200_000;

const MAX_ACC_VALUE: u128 = 1_000_000_000 / 8;

/// How many frames to store for the moving average frame time
const STORED_FRAME_TIME_COUNT: usize = 10;

fn should_snap(dt: u128, target: u128) -> bool {
    if dt < target {
        target - dt < SNAP_THRESHOLD
    } else {
        dt - target < SNAP_THRESHOLD
    }
}

/// Implement this on your main state type to have [`Game`][self::Game]
/// manage the game loop for you.
pub trait GameState: Sized + 'static {
    /// Create the initial state.
    ///
    /// This is called immediately after the game loop is started.
    /// It is done this way due to async functions involved in the creation of the renderer,
    /// which is easiest to handle within a single encompassing async function (especially in wasm).
    fn init(game: &mut Game) -> Self;
    /// Advance the game forward by a timestep of `Game::dt_fixed` seconds. Return None to exit the game.
    fn tick(&mut self, game: &mut Game) -> Option<()>;
    /// Render the game onto the screen. `dt` is the time in seconds since last draw.
    fn draw(&mut self, game: &mut Game, dt: f32);
}

#[derive(Clone, Debug)]
pub struct GameParams<State: GameState> {
    pub window: WindowBuilder,
    pub on_event: fn(&mut State, &Event<()>),
    pub graphics: GraphicsConfig,
}

#[derive(Clone, Copy, Debug)]
pub struct GraphicsConfig {
    pub fps: u32,
    pub use_vsync: bool,
    pub lighting_quality: crate::LightingQualityConfig,
}

impl<State: GameState> Default for GameParams<State> {
    fn default() -> Self {
        Self {
            window: WindowBuilder::new()
                .with_title("starframe")
                .with_inner_size(winit::dpi::LogicalSize {
                    width: 1280.0,
                    height: 720.0,
                }),
            on_event: |_, _| {},
            graphics: GraphicsConfig::default(),
        }
    }
}

impl Default for GraphicsConfig {
    fn default() -> Self {
        Self {
            fps: 60,
            use_vsync: true,
            lighting_quality: crate::LightingQualityConfig::default(),
        }
    }
}

/// A Game manages the global resources a game needs like a window and a graphics renderer
/// and handles timing of the game loop.
pub struct Game {
    /// Current state of input devices.
    pub input: crate::Input,
    /// A manager that handles the rendering context and GPU resources.
    pub renderer: crate::Renderer,
    /// Interface for loading and rendering graphics assets.
    pub graphics: crate::GraphicsManager,
    /// Main ECS world of the game.
    pub world: hecs::World,
    /// Main physics world of the game.
    pub physics: PhysicsWorld,
    /// Handler for interactions between the ECS world and the physics world.
    pub hecs_sync: HecsSyncManager,
    /// Fixed delta-time between frames.
    pub dt_fixed: f64,
    /// Duration of a frame in nanoseconds.
    nanos_per_frame: u128,
    /// Durations of the last N frames to allow displaying a moving average frame time.
    last_frame_times: VecDeque<f32>,
}

/// An error that occurred during in the initialization
/// of a game window and renderer.
#[derive(Debug, thiserror::Error)]
pub enum GameError {
    #[error("Error with winit event loop")]
    EventLoopError(#[from] winit::error::EventLoopError),
    #[error("Failed to create a window")]
    WindowOSError(#[from] winit::error::OsError),
    #[error("Failed to initialize wgpu renderer")]
    RendererInitError(#[from] RendererInitError),
}

impl Game {
    pub fn run<State: GameState>(params: GameParams<State>) -> Result<(), GameError> {
        let events: EventLoop<()> = EventLoop::new()?;
        let window = params.window.build(&events)?;
        #[cfg(not(target_arch = "wasm32"))]
        {
            futures::executor::block_on(Self::run_async(
                params.on_event,
                events,
                window,
                params.graphics,
            ))?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init().expect("Failed to initialize console logger");
            use winit::platform::web::WindowExtWebSys;
            let canvas = web_sys::Element::from(window.canvas());
            web_sys::window()
                .and_then(|win| win.document())
                // made-up convention for putting the game in a specific spot in the DOM
                .and_then(|doc| match doc.get_element_by_id("starframe-game") {
                    Some(parent) => parent.append_child(&canvas).ok(),
                    None => doc.body().and_then(|body| body.append_child(&canvas).ok()),
                })
                .expect("couldn't append canvas to document body");
            wasm_bindgen_futures::spawn_local(Self::run_async(
                params.fps,
                params.on_event,
                events,
                window,
                params.lighting_quality_config,
            ))?;
        }
        Ok(())
    }

    async fn run_async<State: GameState>(
        on_event: fn(&mut State, &Event<()>),
        events: EventLoop<()>,
        window: Window,
        graphics_conf: GraphicsConfig,
    ) -> Result<(), GameError> {
        let _tracy_client = tracy_client::Client::start();

        //
        // init
        //

        let renderer = crate::Renderer::init(window, graphics_conf).await?;

        let mut game = Game {
            input: crate::Input::new(),
            graphics: crate::GraphicsManager::new(),
            renderer,
            world: hecs::World::new(),
            physics: PhysicsWorld::new(
                crate::physics::TuningConstants::default(),
                crate::CollisionMaskMatrix::default(),
            ),
            hecs_sync: HecsSyncManager::new_autosync(crate::HecsSyncOptions::both_ways()),
            nanos_per_frame: 1_000_000_000 / u128::from(graphics_conf.fps),
            dt_fixed: 1.0 / graphics_conf.fps as f64,
            last_frame_times: [1. / graphics_conf.fps as f32; STORED_FRAME_TIME_COUNT]
                .into_iter()
                .collect(),
        };
        let mut state = State::init(&mut game);

        //
        // loop
        //

        let mut frame_start_t = Instant::now();
        // begin the frame time accumulator at nanos_per_frame
        // to cause one frame to be simulated before first draw
        let mut acc = game.nanos_per_frame;
        events.run(move |event, elwt| {
            (on_event)(&mut state, &event);

            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                Event::AboutToWait => {
                    // if vsynced, pretend frame timing is exact (see blog post mentioned above)
                    let dt = frame_start_t.elapsed();
                    let mut dt_nanos = dt.as_nanos();
                    if should_snap(dt_nanos, NANOS_120FPS) {
                        dt_nanos = NANOS_120FPS;
                        acc = 0;
                    } else if should_snap(dt_nanos, NANOS_60FPS) {
                        dt_nanos = NANOS_60FPS;
                        acc = 0;
                    } else if should_snap(dt_nanos, NANOS_30FPS) {
                        dt_nanos = NANOS_30FPS;
                        acc = 0;
                    } else if should_snap(dt_nanos, NANOS_20FPS) {
                        dt_nanos = NANOS_20FPS;
                        acc = 0;
                    } else if should_snap(dt_nanos, NANOS_15FPS) {
                        dt_nanos = NANOS_15FPS;
                        acc = 0;
                    }

                    acc += dt_nanos;
                    frame_start_t = Instant::now();

                    // limit acc to prevent spiral of death
                    if acc > MAX_ACC_VALUE {
                        acc = MAX_ACC_VALUE;
                    }

                    // run gameplay ticks at a constant rate

                    while acc >= game.nanos_per_frame {
                        let _frame = tracy_client::non_continuous_frame!("tick");

                        if state.tick(&mut game).is_none() {
                            elwt.exit();
                            return;
                        }
                        game.input.tick();
                        acc -= game.nanos_per_frame;
                    }

                    // draw as fast as we can

                    let dt_secs = dt.as_secs_f32();
                    {
                        let _draw_span = tracy_client::span!("draw");

                        state.draw(&mut game, dt_secs);
                    }

                    game.last_frame_times.pop_front();
                    game.last_frame_times.push_back(dt_secs);

                    game.renderer.profiler.end_frame().unwrap();
                    tracy_client::frame_mark();

                    game.renderer
                        .profiler
                        .process_finished_frame(crate::Renderer::queue().get_timestamp_period());
                }
                Event::WindowEvent { event, .. } => {
                    game.input.track_window_event(&event);
                    match event {
                        WindowEvent::CloseRequested => {
                            elwt.exit();
                        }
                        WindowEvent::Resized(new_size) => {
                            game.renderer.resize_swap_chain(new_size);
                        }
                        _ => (),
                    }
                }
                _ => (),
            }
        })?;

        Ok(())
    }

    /// Step the game's physics world forward in time by a frame.
    ///
    /// Convenience method that calls [`HecsSyncManager::sync_hecs_to_physics`],
    /// [`PhysicsWorld::tick`], and [`HecsSyncManager::sync_physics_to_hecs`].
    pub fn physics_tick(&mut self, ff: &impl ForceField, time_scale: Option<f64>) {
        self.hecs_sync
            .sync_hecs_to_physics(&mut self.physics, &mut self.world);
        self.physics.tick(self.dt_fixed, time_scale, ff);
        self.hecs_sync
            .sync_physics_to_hecs(&self.physics, &mut self.world);
    }

    /// Clear all state stored in the game struct,
    /// namely `self.graphics`, `self.world`, `self.physics` and `self.hecs_sync`.
    pub fn clear_state(&mut self) {
        self.graphics.clear();
        self.world.clear();
        self.physics.clear();
        self.hecs_sync.clear();
    }

    /// Get the average recent framerate as ms/frame.
    pub fn get_framerate(&self) -> f32 {
        1000. * self.last_frame_times.iter().fold(0., |acc, x| acc + x)
            / self.last_frame_times.len() as f32
    }
}
