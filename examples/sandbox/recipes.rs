//! Types and procedures for reading objects from files and spawning them in the game world.
//!
//! This file has gotten quite large and unwieldy over time.
//! TODO: streamline this and bring in the Tiled editor integration from Flamegrower

use itertools::Itertools;
use sf::math::ConvertPrecision;
use starframe as sf;

use rand::{distributions as distr, distributions::Distribution, Rng};

#[derive(Clone, Debug, serde::Deserialize)]
pub enum Recipe {
    Player(crate::player::PlayerRecipe),
    Block(Block),
    Ball(Ball),
    Capsule(Capsule),
    GenericBody {
        #[serde(with = "sf::serde_pose")]
        pose: sf::Pose,
        colliders: Vec<sf::Collider>,
        is_lit: bool,
    },
    Blockchain {
        width: f64,
        spacing: f64,
        links: Vec<[f64; 2]>,
        anchored_start: bool,
        anchored_end: bool,
    },
    Oscillator {
        position: [f32; 2],
        begin_length: f32,
        target_length: f64,
        compliance: f64,
    },
    RopeConnectedBlocks {
        block1: Block,
        offset1: [f64; 2],
        block2: Block,
        offset2: [f64; 2],
    },
    BackgroundTree {
        #[serde(with = "sf::serde_pose")]
        pose: sf::Pose,
        start_time: f32,
    },
    BackgroundWall {
        left: f32,
        top: f32,
        right: f32,
        bottom: f32,
        depth: f32,
    },
    BackgroundForest {
        mesh_count: usize,
        anim_count: usize,
        left: f32,
        right: f32,
        top: f32,
        bottom: f32,
        front: f32,
        back: f32,
    },
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
pub struct Ball {
    pub radius: f32,
    pub position: [f32; 2],
    pub restitution: f64,
    pub start_velocity: [f64; 2],
    pub is_static: bool,
}

impl Default for Ball {
    fn default() -> Self {
        Self {
            radius: 1.0,
            position: [0.0, 0.0],
            restitution: 0.0,
            start_velocity: [0.0, 0.0],
            is_static: false,
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
pub struct Capsule {
    pub length: f64,
    pub radius: f64,
    pub pose: sf::PoseBuilder,
    pub is_static: bool,
}

impl Default for Capsule {
    fn default() -> Self {
        Self {
            length: 1.0,
            radius: 0.5,
            pose: Default::default(),
            is_static: false,
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
pub struct Block {
    pub width: f64,
    pub height: f64,
    pub radius: f64,
    pub pose: sf::PoseBuilder,
    pub is_static: bool,
}

impl Default for Block {
    fn default() -> Self {
        Self {
            width: 1.0,
            height: 1.0,
            radius: 0.0,
            pose: Default::default(),
            is_static: false,
        }
    }
}

fn spawn_block(game: &mut sf::Game, block: Block) -> sf::hecs::Entity {
    let pose = sf::Pose::from(block.pose);
    let coll = sf::Collider::new_rounded_rect(block.width, block.height, block.radius);
    let coll_key = game.physics.entity_set.insert_collider(coll);
    let mesh_id = game.graphics.create_mesh(sf::MeshParams {
        data: sf::MeshData::from(coll),
        ..Default::default()
    });
    let mat = game.graphics.get_material_id("wall").unwrap();
    game.graphics.set_mesh_material(mesh_id, mat);

    let entity = game.world.spawn((pose, coll_key, mesh_id));

    if !block.is_static {
        let body = sf::Body::new_dynamic(coll.info(), 0.5);
        let body_key = game.physics.entity_set.insert_body(body);
        game.physics
            .entity_set
            .attach_existing_collider(body_key, coll_key);
        game.world.insert_one(entity, body_key).ok();
    }
    entity
}

#[derive(Debug)]
struct Solid<'a> {
    pose: sf::Pose,
    colliders: &'a [sf::Collider],
    material: Option<sf::MaterialId>,
    light_color: Option<[f32; 4]>,
}

fn spawn_static(game: &mut sf::Game, solid: Solid) {
    for coll in solid.colliders {
        let phys_pose = sf::PhysicsPose::from(solid.pose);
        let coll_key = game
            .physics
            .entity_set
            .insert_collider(coll.with_pose(phys_pose));
        let mesh_id = game.graphics.create_mesh(sf::MeshParams {
            data: sf::MeshData::from(*coll),
            offset: solid.pose * sf::Pose::from(coll.pose),
            ..Default::default()
        });
        if let Some(mat_id) = solid.material {
            game.graphics.set_mesh_material(mesh_id, mat_id);
        }
        let ent = game.world.spawn((coll_key, mesh_id));
        if let Some(l_col) = solid.light_color {
            let light = sf::PointLight {
                color: [l_col[0], l_col[1], l_col[2]],
                position: sf::uv::Vec3::zero(),
                ..Default::default()
            };
            game.world.insert_one(ent, light).unwrap();
        }
    }
}

fn spawn_body(game: &mut sf::Game, solid: Solid) -> sf::BodyKey {
    let coll_setup = sf::CompoundColliderSetup::new(solid.colliders);
    let center_of_mass = coll_setup.center_of_mass();

    let body = sf::Body::new_dynamic(coll_setup.info_around_point(center_of_mass), 0.5)
        .with_pose(solid.pose.into());
    let body_key = game.physics.entity_set.insert_body(body);

    for mut coll in solid.colliders.iter().cloned() {
        coll.pose.translation -= center_of_mass;
        let coll_key = game.physics.entity_set.attach_collider(body_key, coll);

        // visualization with a mesh entity synced from physics
        let mesh_id = game.graphics.create_mesh(sf::MeshParams {
            data: sf::MeshData::from(coll),
            ..Default::default()
        });
        if let Some(mat_id) = solid.material {
            game.graphics.set_mesh_material(mesh_id, mat_id);
        }
        let ent = game.world.spawn((solid.pose, coll_key, mesh_id));
        // this is needed for compound colliders,
        // could/should probably be inferred automatically
        // and/or worked around in some other more clean way
        // (maybe merge meshes into one so we just have one entity?)
        game.hecs_sync.register_collider(
            coll_key,
            ent,
            sf::HecsSyncOptions::physics_to_hecs_only(),
        );
        if let Some(l_col) = solid.light_color {
            let light = sf::PointLight {
                color: [l_col[0], l_col[1], l_col[2]],
                position: sf::uv::Vec3::new(0., 0., -2.),
                ..Default::default()
            };
            game.world.insert_one(ent, light).unwrap();
        }
    }

    body_key
}

impl Recipe {
    pub fn spawn(&self, game: &mut sf::Game, gen_assets: &super::GeneratedAssets) {
        let mut rng = rand::thread_rng();
        let mut random_palette = |is_lit: bool| {
            let idx = rng.gen_range(0..super::PALETTE_COLORS.len());
            let mat = if is_lit {
                gen_assets.light_palette[idx]
            } else {
                gen_assets.translucent_palette[idx]
            };
            (super::PALETTE_COLORS[idx], mat)
        };
        match self {
            Recipe::Player(p_rec) => p_rec.spawn(game, gen_assets.player),
            Recipe::Block(block) => {
                spawn_block(game, *block);
            }
            Recipe::Ball(Ball {
                radius,
                position,
                restitution,
                start_velocity,
                is_static,
            }) => {
                let pose = sf::Pose::new(position.into(), sf::Angle::default());
                let coll =
                    sf::Collider::new_circle(*radius as f64).with_material(sf::PhysicsMaterial {
                        restitution_coef: *restitution,
                        ..Default::default()
                    });
                let (col, mat) = random_palette(false);
                let solid = Solid {
                    pose,
                    colliders: &mut [coll],
                    material: Some(mat),
                    light_color: Some(col),
                };
                if *is_static {
                    spawn_static(game, solid);
                } else {
                    let body_key = spawn_body(game, solid);
                    let body = game.physics.entity_set.get_body_mut(body_key).unwrap();
                    body.velocity.linear = start_velocity.into();
                }
            }
            Recipe::Capsule(Capsule {
                length,
                radius,
                pose,
                is_static,
            }) => {
                let (col, mat) = random_palette(false);
                let solid = Solid {
                    pose: (*pose).into(),
                    colliders: &mut [sf::Collider::new_capsule(*length, *radius)],
                    material: Some(mat),
                    light_color: Some(col),
                };
                if *is_static {
                    spawn_static(game, solid);
                } else {
                    spawn_body(game, solid);
                }
            }
            Recipe::GenericBody {
                pose,
                colliders,
                is_lit,
            } => {
                let (col, mat) = random_palette(*is_lit);
                let solid = Solid {
                    pose: *pose,
                    colliders,
                    material: Some(mat),
                    light_color: Some(col),
                };
                spawn_body(game, solid);
            }
            Recipe::Blockchain {
                width,
                spacing,
                links,
                anchored_start,
                anchored_end,
            } => {
                if links.len() < 2 {
                    println!("Too few links in a chain");
                    return;
                }

                let half_spacing = spacing / 2.0;
                let radius = width / 2.0;

                let mut links_iter = links.iter().map(|p| sf::DVec2::new(p[0], p[1])).peekable();

                // to connect another block to it
                let mut prev_block: Option<(sf::BodyKey, f64)> = None;
                while let (Some(link1), Some(link2)) = (links_iter.next(), links_iter.peek()) {
                    let distance = *link2 - link1;
                    let dist_norm = distance.mag();
                    let center = (link1 + *link2) / 2.0;
                    let orientation = (distance[0] / dist_norm).acos() * distance[1].signum();

                    let caps_full_length = dist_norm - spacing;
                    let (col, mat) = random_palette(false);
                    let capsule = spawn_body(
                        game,
                        Solid {
                            pose: sf::PoseBuilder::new()
                                .with_position(center.conv_p())
                                .with_rotation(sf::Angle::Rad(orientation as f32))
                                .into(),
                            colliders: &mut [sf::Collider::new_capsule(
                                caps_full_length - width,
                                radius,
                            )],
                            material: Some(mat),
                            light_color: Some(col),
                        },
                    );
                    let caps_length_half = caps_full_length / 2.0;
                    if let Some((prev_block, prev_block_offset)) = prev_block {
                        game.physics.constraint_set.insert(
                            sf::ConstraintBuilder::new(capsule)
                                .with_target(prev_block)
                                .with_origin(sf::DVec2::new(-caps_length_half - half_spacing, 0.0))
                                .with_target_origin(sf::DVec2::new(prev_block_offset, 0.0))
                                .with_compliance(0.015)
                                .build_attachment(),
                        );
                    } else if *anchored_start {
                        game.physics.constraint_set.insert(
                            sf::ConstraintBuilder::new(capsule)
                                .with_origin(sf::DVec2::new(-caps_length_half - half_spacing, 0.0))
                                .with_target_origin(link1)
                                .build_attachment(),
                        );
                    }
                    prev_block = Some((capsule, caps_length_half + half_spacing));
                }

                if *anchored_end {
                    let (prev_block, prev_block_offset) = prev_block.unwrap();
                    game.physics.constraint_set.insert(
                        sf::ConstraintBuilder::new(prev_block)
                            .with_origin(sf::DVec2::new(prev_block_offset + (spacing / 2.0), 0.0))
                            .with_target_origin(
                                links
                                    .iter()
                                    .map(|p| sf::DVec2::new(p[0], p[1]))
                                    .last()
                                    .unwrap(),
                            )
                            .build_attachment(),
                    );
                }
            }
            Recipe::Oscillator {
                position,
                begin_length,
                target_length,
                compliance,
            } => {
                let position: sf::Vec2 = position.into();
                let offset = sf::Vec2::new(begin_length / 2.0, 0.0);
                let b1 = spawn_block(
                    game,
                    Block {
                        width: 1.0,
                        height: 1.0,
                        radius: 0.0,
                        pose: sf::PoseBuilder::new().with_position(position + offset),
                        is_static: false,
                    },
                );
                let b1 = *game.world.query_one_mut::<&sf::BodyKey>(b1).unwrap();
                let b2 = spawn_block(
                    game,
                    Block {
                        width: 1.0,
                        height: 1.0,
                        radius: 0.0,
                        pose: sf::PoseBuilder::new().with_position(position - offset),
                        is_static: false,
                    },
                );
                let b2 = *game.world.query_one_mut::<&sf::BodyKey>(b2).unwrap();
                game.physics.constraint_set.insert(
                    sf::ConstraintBuilder::new(b1)
                        .with_target(b2)
                        .with_compliance(*compliance)
                        .with_linear_damping(0.0)
                        .build_distance(*target_length),
                );
            }
            Recipe::RopeConnectedBlocks {
                block1,
                offset1,
                block2,
                offset2,
            } => {
                let b1 = spawn_block(game, *block1);
                let b1 = game.world.query_one_mut::<&sf::BodyKey>(b1).copied();
                let b2 = spawn_block(game, *block2);
                let b2 = game.world.query_one_mut::<&sf::BodyKey>(b2).copied();
                let rope_end_1 = (block1.pose.build() * sf::DVec2::from(offset1).conv_p()).conv_p();
                let rope_end_2 = (block2.pose.build() * sf::DVec2::from(offset2).conv_p()).conv_p();
                let rope = sf::Rope::spawn_line(
                    sf::RopeParameters {
                        ..Default::default()
                    },
                    rope_end_1,
                    rope_end_2,
                    &mut game.physics.entity_set,
                );
                // temporary visualisation with individual particle Meshes
                let mesh_id = game.graphics.create_mesh(sf::MeshParams {
                    data: sf::MeshData::from(sf::ConvexMeshShape::Circle {
                        r: rope.params.thickness / 2.0,
                        points: 8,
                    }),
                    ..Default::default()
                });
                let rope_mat = game.graphics.get_material_id("wall").unwrap();
                game.graphics.set_mesh_material(mesh_id, rope_mat);
                for particle in &rope.particles {
                    let mesh_ent = game.world.spawn((sf::Pose::default(), mesh_id, *particle));
                    game.hecs_sync.register_body(
                        particle.body,
                        mesh_ent,
                        sf::HecsSyncOptions::physics_to_hecs_only(),
                    );
                }
                let first_particle = *rope.particles.first().expect("No particles in rope");
                let last_particle = *rope.particles.iter().last().expect("No particles in rope");
                match b1 {
                    Ok(b1) => {
                        game.physics.constraint_set.insert(
                            sf::ConstraintBuilder::new(first_particle.body)
                                .with_target(b1)
                                .with_target_origin(offset1.into())
                                .build_attachment(),
                        );
                    }
                    Err(_) => {
                        game.physics.constraint_set.insert(
                            sf::ConstraintBuilder::new(first_particle.body)
                                .with_target_origin(rope_end_1)
                                .build_attachment(),
                        );
                    }
                }
                match b2 {
                    Ok(b2) => {
                        game.physics.constraint_set.insert(
                            sf::ConstraintBuilder::new(last_particle.body)
                                .with_target(b2)
                                .with_target_origin(offset2.into())
                                .build_attachment(),
                        );
                    }
                    Err(_) => {
                        game.physics.constraint_set.insert(
                            sf::ConstraintBuilder::new(last_particle.body)
                                .with_target_origin(rope_end_2)
                                .build_attachment(),
                        );
                    }
                }
                game.physics.rope_set.insert(rope);
            }
            Recipe::BackgroundWall {
                left,
                top,
                right,
                bottom,
                depth,
            } => {
                let mesh_id = game.graphics.create_mesh(sf::MeshParams {
                    // TODO: ConvexMeshShape should take f32s
                    data: sf::MeshData::from(sf::ConvexMeshShape::Rect {
                        w: (right - left) as f64,
                        h: (top - bottom) as f64,
                    }),
                    ..Default::default()
                });

                let pose = sf::PoseBuilder::new()
                    .with_position([(right + left) / 2., (top + bottom) / 2.])
                    .with_depth(*depth)
                    .build();

                game.world.spawn((mesh_id, pose));
            }
            Recipe::BackgroundTree { pose, start_time } => {
                let mesh_id = game.graphics.get_mesh_id("library.tree_mesh").unwrap();
                let mesh_id = game.graphics.new_animation_target(mesh_id);
                let mut anim =
                    sf::Animator::new(game.graphics.get_animation_id("library.sway").unwrap())
                        .with_target(mesh_id);
                anim.t = *start_time;
                game.graphics.insert_animator(anim);
                game.world.spawn((*pose, mesh_id));
            }
            Recipe::BackgroundForest {
                mesh_count,
                anim_count,
                left,
                right,
                top,
                bottom,
                front,
                back,
            } => {
                // spawn a ton of trees with a few shared animation states
                let anim_id = game.graphics.get_animation_id("library.sway").unwrap();
                let mesh_id = game.graphics.get_mesh_id("library.tree_mesh").unwrap();

                let targets: Vec<sf::MeshId> = std::iter::once(mesh_id)
                    .chain(std::iter::repeat_with(|| {
                        game.graphics.new_animation_target(mesh_id)
                    }))
                    .take(*anim_count)
                    .collect_vec();

                let time_increment = 4. / targets.len() as f32;
                for (i, target) in targets.iter().enumerate() {
                    let mut animator = sf::Animator::new(anim_id).with_target(*target);
                    animator.t = time_increment * i as f32;
                    game.graphics.insert_animator(animator);
                }

                for mesh_idx in 0..*mesh_count {
                    let pose = sf::PoseBuilder::new()
                        .with_position([
                            distr::Uniform::from(*left..*right).sample(&mut rng),
                            distr::Uniform::from(*bottom..*top).sample(&mut rng),
                        ])
                        .with_depth(distr::Uniform::from(*front..*back).sample(&mut rng))
                        .build();
                    game.world.spawn((pose, targets[mesh_idx % targets.len()]));
                }
            }
        }
    }
}
