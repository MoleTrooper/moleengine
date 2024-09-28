#[cfg(feature = "tracy")]
#[global_allocator]
static GLOBAL: tracy_client::ProfiledAllocator<std::alloc::System> =
    tracy_client::ProfiledAllocator::new(std::alloc::System, 100);

use itertools::izip;
use rand::{distributions as distr, distributions::Distribution};

use starframe as sf;

mod mousegrab;
use mousegrab::MouseGrabber;
mod player;
mod recipes;
use recipes::Recipe;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let window = winit::window::WindowBuilder::new()
        .with_title("starframe test")
        .with_inner_size(winit::dpi::LogicalSize {
            width: 1280.0,
            height: 720.0,
        });

    sf::Game::run(sf::GameParams {
        window,
        on_event: |state: &mut State, evt| {
            if let winit::event::Event::WindowEvent { event, .. } = evt {
                let egui_resp = state
                    .egui_state
                    .on_window_event(sf::Renderer::window(), event);
                if egui_resp.consumed {
                    // TODO: don't propagate the event
                }
            }
        },
        graphics: sf::GraphicsConfig {
            fps: 60,
            use_vsync: std::env::var("NO_VSYNC").is_err(),
            lighting_quality: sf::LightingQualityConfig::default(),
        },
    })?;

    Ok(())
}

//
// Types
//

/// smooth interpolation between environment map presets, just for fun
#[derive(Clone)]
enum EnvironmentMapState {
    Static(sf::EnvironmentMap),
    Interpolating {
        start: sf::EnvironmentMap,
        end: sf::EnvironmentMap,
        t: f32,
    },
}

pub enum StateEnum {
    Playing,
    Paused,
}
pub struct State {
    scenes_available: Vec<std::path::PathBuf>,
    scene: Scene,
    state: StateEnum,
    // gameplay
    mouse_grabber: MouseGrabber,
    // graphics
    camera: sf::Camera,
    env_map: EnvironmentMapState,
    light_quality: sf::LightingQualityConfig,
    gen_assets: GeneratedAssets,
    camera_ctl: sf::MouseDragCameraController,
    // egui stuff
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    last_egui_output: egui::FullOutput,
    // UI states
    bvh_vis_active: bool,
    bvh_vis_levels: usize,
    island_vis_active: bool,
    spawner_circle_r: f64,
    spawner_obj_count: usize,
    spawner_is_lit: bool,
    time_scale: f64,
}
impl State {
    fn init(game: &mut sf::Game) -> Self {
        let gen_assets = load_common_assets(game);

        let egui_context = egui::Context::default();
        let viewport_id = egui_context.viewport_id();
        State {
            scenes_available: read_available_scenes().expect("Failed to read scenes directory"),
            scene: Scene::default(),
            state: StateEnum::Playing,
            mouse_grabber: MouseGrabber::new(),
            camera: sf::Camera::default(),
            // default environment map simulates a soft moonlight
            // so that dynamic lights inside of the scene look bright
            env_map: EnvironmentMapState::Static(sf::EnvironmentMap::preset_night()),
            light_quality: sf::LightingQualityConfig::default(),
            gen_assets,
            camera_ctl: sf::MouseDragCameraController {
                activate_button: sf::MouseButton::Middle.into(),
                reset_button: Some(sf::Key::KeyR.into()),
                ..Default::default()
            },
            egui_state: egui_winit::State::new(
                egui_context,
                viewport_id,
                sf::Renderer::window(),
                None,
                None,
            ),
            egui_renderer: egui_wgpu::Renderer::new(
                sf::Renderer::device(),
                game.renderer.swapchain_format(),
                Some(game.renderer.depth_format()),
                sf::graphics::renderer::MSAA_SAMPLES,
            ),
            last_egui_output: Default::default(),
            bvh_vis_active: false,
            bvh_vis_levels: 30,
            island_vis_active: false,
            spawner_circle_r: 0.0,
            spawner_obj_count: 1,
            spawner_is_lit: true,
            time_scale: 1.0,
        }
    }
}

/// Set of colors to pick from for randomly spawned objects
const PALETTE_COLORS: [[f32; 4]; 6] = [
    [0.910, 0.582, 0.582, 1.],
    [0.813, 0.910, 0.546, 1.],
    [0.904, 0.910, 0.546, 1.],
    [0.696, 0.940, 0.936, 1.],
    [0.836, 0.721, 0.890, 1.],
    [0.890, 0.721, 0.851, 1.],
];

pub struct GeneratedAssets {
    player: player::PlayerMeshes,
    light_palette: Vec<sf::MaterialId>,
    translucent_palette: Vec<sf::MaterialId>,
}

/// Load assets referenced by name elsewhere.
///
/// Currently, this must be called after [`State::reset`] before loading a level.
/// It would be nice to have a form of garbage collection for `GraphicsManager`
/// that doesn't remove these, but that's not a top priority right now
fn load_common_assets(game: &mut sf::Game) -> GeneratedAssets {
    game.graphics
        .load_gltf("examples/sandbox/assets/library.glb")
        .expect("Failed to load shared assets");

    let player = player::controller::upload_meshes(&mut game.graphics);

    let light_palette = PALETTE_COLORS
        .into_iter()
        .map(|col| {
            game.graphics.create_material(
                sf::MaterialParams {
                    base_color: Some(col),
                    emissive_color: Some(col),
                    attenuation: Some(sf::AttenuationParams {
                        color: [col[0], col[1], col[2]],
                        distance: 1.,
                    }),
                    ..Default::default()
                },
                None,
            )
        })
        .collect();

    let translucent_palette = PALETTE_COLORS
        .into_iter()
        .map(|col| {
            game.graphics.create_material(
                sf::MaterialParams {
                    base_color: Some(col),
                    attenuation: Some(sf::AttenuationParams {
                        color: [col[0], col[1], col[2]],
                        distance: 0.05,
                    }),
                    ..Default::default()
                },
                None,
            )
        })
        .collect();

    game.graphics.create_material(
        sf::MaterialParams {
            base_color: Some([0.5, 0.5, 0.5, 1.]),
            attenuation: Some(sf::AttenuationParams {
                color: [0.1; 3],
                distance: 0.5,
            }),
            ..Default::default()
        },
        Some("wall"),
    );

    GeneratedAssets {
        player,
        light_palette,
        translucent_palette,
    }
}

//
// scenes & loading
//

/// The recipes in a scene plus some adjustable parameters.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct Scene {
    gravity: [f64; 2],
    spawn_zone: sf::AABB,
    recipes: Vec<Recipe>,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            gravity: [0.0, -9.81],
            spawn_zone: sf::AABB {
                min: sf::uv::DVec2::new(-5.0, 1.0),
                max: sf::uv::DVec2::new(5.0, 4.0),
            },
            recipes: vec![],
        }
    }
}

impl Scene {
    pub fn read_from_file(file: std::fs::File) -> Result<Self, ron::de::Error> {
        use serde::Deserialize;
        use std::io::Read;

        let mut reader = std::io::BufReader::new(file);
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;

        let mut deser = ron::de::Deserializer::from_bytes(bytes.as_slice())?;
        Scene::deserialize(&mut deser)
    }

    pub fn read_from_string(s: &str) -> Result<Self, ron::de::Error> {
        let mut deser = ron::de::Deserializer::from_str(s)?;
        <Self as serde::Deserialize>::deserialize(&mut deser)
    }

    pub fn instantiate(&self, game: &mut sf::Game, gen_assets: &GeneratedAssets) {
        for recipe in &self.recipes {
            recipe.spawn(game, gen_assets);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn read_available_scenes() -> Result<Vec<std::path::PathBuf>, std::io::Error> {
    let dir = std::fs::read_dir("./examples/sandbox/scenes")?;
    Ok(dir
        .into_iter()
        .filter_map(|entry| entry.map(|e| e.path()).ok())
        .filter(|p| p.is_file())
        .collect())
}

#[cfg(not(target_arch = "wasm32"))]
fn read_scene(path: &std::path::Path) -> Option<Scene> {
    let file = std::fs::File::open(path);
    match file {
        Ok(file) => match Scene::read_from_file(file) {
            Ok(scene) => Some(scene),
            Err(err) => {
                eprintln!("Failed to parse file: {}", err);
                None
            }
        },
        Err(err) => {
            eprintln!("Failed to open file: {}", err);
            None
        }
    }
}

// hackery to simulate loading scenes with include_str on the web
// so I don't have to implement downloading stuff at runtime
#[cfg(target_arch = "wasm32")]
fn read_available_scenes() -> Result<Vec<std::path::PathBuf>, std::io::Error> {
    Ok([
        "compound_colliders.ron",
        "minimal.ron",
        "oscillators.ron",
        "ropes.ron",
        "test.ron",
    ]
    .into_iter()
    .map(|s| std::path::PathBuf::from(s))
    .collect())
}

#[cfg(target_arch = "wasm32")]
fn read_scene(path: &std::path::Path) -> Option<Scene> {
    let scene_str = match path.to_str().unwrap() {
        "compound_colliders.ron" => include_str!("scenes/compound_colliders.ron"),
        "minimal.ron" => include_str!("scenes/minimal.ron"),
        "oscillators.ron" => include_str!("scenes/oscillators.ron"),
        "ropes.ron" => include_str!("scenes/ropes.ron"),
        "test.ron" => include_str!("scenes/test.ron"),
        _ => {
            log::error!("Scene not included: {path:?}");
            return None;
        }
    };
    match Scene::read_from_string(scene_str) {
        Ok(scene) => Some(scene),
        Err(err) => {
            log::error!("Failed to parse scene: {err}");
            None
        }
    }
}

//
// State updates
//

impl sf::GameState for State {
    fn init(game: &mut sf::Game) -> Self {
        Self::init(game)
    }

    fn tick(&mut self, game: &mut sf::Game) -> Option<()> {
        let mut rng = rand::thread_rng();

        //
        // gui controls
        //

        let egui_input = self.egui_state.take_egui_input(sf::Renderer::window());
        self.egui_state.egui_ctx().begin_frame(egui_input);

        let mut exit = false;
        let mut reload = false;
        let mut step_one = false;
        let mut shape_to_spawn: Option<sf::ColliderPolygon> = None;
        let mut light_quality = self.light_quality;
        let current_env_map = match &self.env_map {
            EnvironmentMapState::Static(m) => m,
            EnvironmentMapState::Interpolating { end, .. } => end,
        };
        let mut next_env_map = current_env_map.clone();

        egui::Window::new("Controls").show(self.egui_state.egui_ctx(), |ui| {
            let framerate = game.get_framerate();
            ui.label(format!("{framerate:.1} ms/frame"));

            ui.separator();

            ui.heading("Load a scene");
            ui.horizontal_wrapped(|ui| {
                for scene_path in &self.scenes_available {
                    if ui
                        .button(
                            scene_path
                                .file_stem()
                                .expect("File with no name?")
                                .to_str()
                                .expect("File with invalid name?"),
                        )
                        .clicked()
                    {
                        if let Some(scene) = read_scene(scene_path) {
                            self.scene = scene;
                            reload = true;
                        }
                    }
                }
            });
            reload |= ui.button("Reload current").clicked();

            ui.separator();
            ui.heading("Spawn objects");
            ui.add(
                egui::Slider::new(&mut self.spawner_obj_count, 1..=50)
                    .text("Number of objects to spawn"),
            );
            ui.add(egui::Slider::new(&mut self.spawner_circle_r, 0.0..=1.0).text("Radius"));
            ui.horizontal_wrapped(|ui| {
                if ui.button("Triangle").clicked() {
                    shape_to_spawn = Some(sf::ColliderPolygon::Triangle {
                        outer_r: distr::Uniform::from(0.5..0.8).sample(&mut rng),
                    });
                }
                if ui.button("Rect").clicked() {
                    shape_to_spawn = Some(sf::ColliderPolygon::Rect {
                        hw: distr::Uniform::from(0.4..0.6).sample(&mut rng),
                        hh: distr::Uniform::from(0.3..0.5).sample(&mut rng),
                    });
                }
                if ui.button("Hexagon").clicked() {
                    shape_to_spawn = Some(sf::ColliderPolygon::Hexagon {
                        outer_r: distr::Uniform::from(0.4..0.7).sample(&mut rng),
                    });
                }
            });
            if self.spawner_circle_r > 0.0 {
                ui.horizontal(|ui| {
                    if ui.button("Circle").clicked() {
                        shape_to_spawn = Some(sf::ColliderPolygon::Point);
                    }
                    if ui.button("Capsule").clicked() {
                        shape_to_spawn = Some(sf::ColliderPolygon::LineSegment {
                            hl: distr::Uniform::from(0.3..0.5).sample(&mut rng),
                        });
                    }
                });
            }
            ui.checkbox(&mut self.spawner_is_lit, "Lit");

            ui.separator();
            ui.heading("Lighting");

            ui.horizontal(|ui| {
                ui.radio_value(
                    &mut light_quality,
                    sf::LightingQualityConfig::ULTRA,
                    "Ultra",
                );
                ui.radio_value(&mut light_quality, sf::LightingQualityConfig::HIGH, "High");
                ui.radio_value(
                    &mut light_quality,
                    sf::LightingQualityConfig::MEDIUM,
                    "Medium",
                );
                ui.radio_value(&mut light_quality, sf::LightingQualityConfig::LOW, "Low");
                ui.radio_value(
                    &mut light_quality,
                    sf::LightingQualityConfig::LOWEST,
                    "Lowest",
                );
            });
            ui.add(
                egui::Slider::new(&mut light_quality.mip_bias, -1.0..=3.0)
                    .step_by(0.5)
                    .text("Mip bias"),
            );

            ui.horizontal(|ui| {
                ui.label("Presets");
                if ui.button("night").clicked() {
                    next_env_map = sf::EnvironmentMap::preset_night();
                }
                if ui.button("sunset").clicked() {
                    next_env_map = sf::EnvironmentMap::preset_sunset();
                }
                if ui.button("day").clicked() {
                    next_env_map = sf::EnvironmentMap::preset_day();
                }
                if ui.button("none").clicked() {
                    next_env_map = sf::EnvironmentMap {
                        // add a light with zero so that interpolation still works
                        lights: vec![sf::DirectionalLight {
                            color: [0.; 3],
                            ..Default::default()
                        }],
                        ..Default::default()
                    };
                }
            });

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label("Ambient");
                    ui.color_edit_button_rgb(&mut next_env_map.ambient);
                });
                ui.vertical(|ui| {
                    ui.label("Sun");
                    ui.color_edit_button_rgb(&mut next_env_map.lights[0].color);
                });
                ui.vertical(|ui| {
                    ui.label("Zenith");
                    ui.color_edit_button_rgb(&mut next_env_map.zenith);
                });
                ui.vertical(|ui| {
                    ui.label("Horizon");
                    ui.color_edit_button_rgb(&mut next_env_map.horizon);
                });
                ui.vertical(|ui| {
                    ui.label("Ground");
                    ui.color_edit_button_rgb(&mut next_env_map.ground);
                });
            });
            ui.add(
                egui::Slider::new(&mut next_env_map.lights[0].direction.x, -1.0..=1.0)
                    .text("Sun direction x"),
            );
            ui.add(
                egui::Slider::new(&mut next_env_map.lights[0].direction.y, -1.0..=1.0)
                    .text("Sun direction y"),
            );

            ui.separator();

            ui.heading("Other controls");
            ui.horizontal(|ui| {
                match self.state {
                    StateEnum::Playing => {
                        if ui.button("Pause").clicked() {
                            self.state = StateEnum::Paused;
                        }
                    }
                    StateEnum::Paused => {
                        if ui.button("Unpause").clicked() {
                            self.state = StateEnum::Playing;
                        }
                    }
                }
                if let StateEnum::Paused = self.state {
                    if ui.button("Step a frame").clicked() {
                        step_one = true;
                    }
                }
            });
            ui.add(
                egui::Slider::new(
                    &mut self.camera.zoom,
                    self.camera_ctl.min_zoom..=self.camera_ctl.max_zoom,
                )
                .text("Camera zoom"),
            );
            ui.add(
                egui::Slider::new(&mut game.physics.consts.substeps, 1..=15)
                    .text("Physics substeps"),
            );
            ui.add(egui::Slider::new(&mut self.time_scale, 0.05..=2.0).text("Time scale"));

            ui.separator();
            ui.heading("Visuals");
            ui.checkbox(&mut self.bvh_vis_active, "Display BVH");
            if self.bvh_vis_active {
                ui.add(
                    egui::Slider::new(&mut self.bvh_vis_levels, 0..=50).text("Tree levels to show"),
                );
            }
            ui.checkbox(&mut self.island_vis_active, "Display islands");
            ui.separator();
            if ui.button("exit").clicked() {
                exit = true;
            }
        });
        if exit {
            return None;
        }
        if reload {
            game.clear_state();
            self.gen_assets = load_common_assets(game);
            self.scene.instantiate(game, &self.gen_assets);
        }

        if light_quality != self.light_quality {
            game.renderer.set_lighting_quality(light_quality);
            self.light_quality = light_quality;
        }

        if next_env_map != *current_env_map {
            self.env_map = EnvironmentMapState::Interpolating {
                start: current_env_map.clone(),
                end: next_env_map,
                t: 0.,
            };
        }

        self.last_egui_output = self.egui_state.egui_ctx().end_frame();

        // mouse controls

        self.mouse_grabber
            .update(&game.input, &self.camera, &mut game.physics);
        self.camera_ctl.update(&mut self.camera, &game.input);

        // spawn stuff even when paused

        let zone = self.scene.spawn_zone;
        let random_pos = || {
            let mut rng = rand::thread_rng();
            sf::Vec2::new(
                distr::Uniform::from(zone.min.x..zone.max.x).sample(&mut rng) as f32,
                distr::Uniform::from(zone.min.y..zone.max.y).sample(&mut rng) as f32,
            )
        };
        let random_angle =
            || sf::Angle::Deg(distr::Uniform::from(0.0..360.0).sample(&mut rand::thread_rng()));

        for _ in 0..self.spawner_obj_count {
            if let Some(polygon) = shape_to_spawn {
                Recipe::GenericBody {
                    pose: sf::PoseBuilder::new()
                        .with_position(random_pos())
                        .with_rotation(random_angle())
                        .build(),
                    colliders: vec![sf::ColliderShape {
                        polygon,
                        circle_r: self.spawner_circle_r,
                    }
                    .into()],
                    is_lit: self.spawner_is_lit,
                }
                .spawn(game, &self.gen_assets);
            }
        }

        match (&self.state, step_one) {
            //
            // Playing or stepping manually
            //
            (StateEnum::Playing, _) | (StateEnum::Paused, true) => {
                if game.input.button(sf::Key::KeyP.into()) {
                    self.state = StateEnum::Paused;
                    return Some(());
                }

                let grav = sf::forcefield::Gravity(self.scene.gravity.into());
                game.physics_tick(&grav, Some(self.time_scale));
                player::controller::tick(game, &self.gen_assets.player);

                Some(())
            }
            //
            // Paused
            //
            (StateEnum::Paused, false) => {
                if game.input.button(sf::Key::KeyP.into()) {
                    self.state = StateEnum::Playing;
                    return Some(());
                }

                Some(())
            }
        }
    }

    fn draw(&mut self, game: &mut sf::Game, dt: f32) {
        let device = sf::Renderer::device();
        let queue = sf::Renderer::queue();
        let window_size = game.renderer.window_size();

        // state updates

        self.camera.upload();

        if matches!(self.state, StateEnum::Playing) {
            game.graphics.update_animations(dt);
        }

        if let EnvironmentMapState::Interpolating { t, end, .. } = &mut self.env_map {
            *t += dt;
            if *t >= 1. {
                self.env_map = EnvironmentMapState::Static(end.clone());
            }
        }

        let env_map = match &self.env_map {
            EnvironmentMapState::Static(m) => m.clone(),
            EnvironmentMapState::Interpolating { start, end, t } => {
                fn smoothstep(t: f32) -> f32 {
                    t * t * (3.0 - 2.0 * t)
                }
                let t = smoothstep(*t);
                let lerp_color = |start: [f32; 3], end: [f32; 3]| -> [f32; 3] {
                    std::array::from_fn(|i| (1. - t) * start[i] + t * end[i])
                };
                sf::EnvironmentMap {
                    ambient: lerp_color(start.ambient, end.ambient),
                    horizon: lerp_color(start.horizon, end.horizon),
                    zenith: lerp_color(start.zenith, end.zenith),
                    ground: lerp_color(start.ground, end.ground),
                    lights: izip!(&start.lights, &end.lights)
                        .map(|(s, e)| sf::DirectionalLight {
                            color: lerp_color(s.color, e.color),
                            direction: (1. - t) * s.direction + t * e.direction,
                        })
                        .collect(),
                }
            }
        };

        game.renderer.set_environment_map(&env_map);

        // scene rendering

        let mut frame = game.renderer.begin_frame();
        frame.set_clear_color([0.00802, 0.0137, 0.02732, 1.]);
        frame.draw_meshes(&mut game.graphics, &mut game.world, &self.camera);

        // egui

        let paint_jobs = self.egui_state.egui_ctx().tessellate(
            self.last_egui_output.shapes.clone(),
            self.egui_state.egui_ctx().pixels_per_point(),
        );
        self.egui_state.handle_platform_output(
            sf::Renderer::window(),
            self.last_egui_output.platform_output.clone(),
        );

        for (tex_id, img_delta) in &self.last_egui_output.textures_delta.set {
            self.egui_renderer
                .update_texture(device, queue, *tex_id, img_delta);
        }

        for tex_id in &self.last_egui_output.textures_delta.free {
            self.egui_renderer.free_texture(tex_id);
        }

        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [window_size.width, window_size.height],
            pixels_per_point: self.egui_state.egui_ctx().pixels_per_point(),
        };
        self.egui_renderer.update_buffers(
            device,
            queue,
            frame.encoder_mut(),
            &paint_jobs,
            &screen_desc,
        );

        {
            let mut pass = frame.pass();
            self.egui_renderer
                .render(&mut pass, &paint_jobs, &screen_desc);
        }
    }
}
