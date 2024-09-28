use starframe as sf;

const MAX_SIMULTANEOUS_BULLETS: usize = 5;
const R: f64 = 0.1;
const LENGTH: f64 = 0.2;
const BASE_MOVE_SPEED: f64 = 4.;
const MAX_ACCELERATION: f64 = 8.;

#[derive(Clone, Debug)]
pub struct PlayerState {
    facing: Facing,
    active_bullets: Vec<(hecs::Entity, sf::ColliderKey)>,
}
impl PlayerState {
    fn new() -> Self {
        PlayerState {
            facing: Facing::Left,
            active_bullets: Vec::with_capacity(MAX_SIMULTANEOUS_BULLETS),
        }
    }
}
#[derive(Clone, Copy, Debug)]
enum Facing {
    Right,
    Left,
}
impl Facing {
    fn orient_vec(&self, vel: sf::Vec2) -> sf::Vec2 {
        match self {
            Facing::Right => vel,
            Facing::Left => sf::Vec2::new(-vel.x, vel.y),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, serde::Deserialize)]
#[serde(default)]
pub struct PlayerRecipe {
    pub position: [f32; 2],
}

impl PlayerRecipe {
    pub fn spawn(&self, game: &mut sf::Game, meshes: PlayerMeshes) {
        let body = sf::Body::new_particle(1.0);
        let body_key = game.physics.entity_set.insert_body(body);
        let coll = player_collider();
        let coll_key = game.physics.entity_set.attach_collider(body_key, coll);
        let pose = sf::PoseBuilder::new()
            .with_position(self.position)
            .with_rotation(sf::Angle::Deg(90.0))
            .build();
        let state = PlayerState::new();

        game.world
            .spawn((body_key, coll_key, pose, meshes.player, state));
    }
}

fn player_collider() -> sf::Collider {
    sf::Collider::new_capsule(LENGTH, R).with_material(sf::PhysicsMaterial {
        static_friction_coef: None,
        dynamic_friction_coef: None,
        restitution_coef: 0.0,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct PlayerMeshes {
    player: sf::MeshId,
    bullet: sf::MeshId,
}

pub mod controller {
    use sf::math::ConvertPrecision;

    use super::*;

    pub fn upload_meshes(graphics: &mut sf::GraphicsManager) -> PlayerMeshes {
        let player = graphics.create_mesh(sf::MeshParams {
            data: sf::MeshData::from(player_collider()),
            name: Some("player"),
            ..Default::default()
        });

        let bullet = graphics.create_mesh(sf::MeshParams {
            data: sf::MeshData::from(sf::ConvexMeshShape::Circle { r: R, points: 5 }),
            name: Some("bullet"),
            ..Default::default()
        });

        PlayerMeshes { player, bullet }
    }

    pub fn tick(game: &mut sf::Game, meshes: &PlayerMeshes) {
        let move_axis = game.input.axis(sf::AxisQuery {
            pos_btn: sf::Key::ArrowRight.into(),
            neg_btn: sf::Key::ArrowLeft.into(),
        });
        let (target_facing, target_hdir) = if move_axis == 0.0 {
            (None, 0.0)
        } else if move_axis.is_sign_positive() {
            (Some(Facing::Right), 1.0)
        } else {
            (Some(Facing::Left), -1.0)
        };

        let mut bullet_delete_queue: Vec<hecs::Entity> = Vec::new();
        let mut bullet_to_spawn = None;

        for (player_entity, (body_key, pose, state)) in
            game.world
                .query_mut::<(&sf::BodyKey, &sf::Pose, &mut PlayerState)>()
        {
            let body = game
                .physics
                .entity_set
                .get_body_mut(*body_key)
                .expect("Player body was unexpectedly deleted");
            // move and orient

            if let Some(facing) = target_facing {
                state.facing = facing;
            }

            let move_speed = BASE_MOVE_SPEED;

            let target_hvel = target_hdir * move_speed;
            let accel_needed = target_hvel - body.velocity.linear.x;
            let accel = accel_needed.min(MAX_ACCELERATION);
            body.velocity.linear.x += accel;

            // jump

            if game.input.button(sf::Key::ShiftLeft.into()) {
                // TODO: only on ground, double jump, custom curve
                body.velocity.linear.y = 4.0;
            }

            // delete bullets that collided with something

            state.active_bullets.retain(|(entity, coll_key)| {
                if game
                    .physics
                    .contacts_for_collider(*coll_key)
                    .next()
                    .is_none()
                {
                    true
                } else {
                    bullet_delete_queue.push(*entity);
                    false
                }
            });

            // shoot

            if state.active_bullets.len() < MAX_SIMULTANEOUS_BULLETS
                && game.input.button(sf::Key::KeyZ.into())
            {
                const R: f64 = 0.05;
                let player_pos = pose.position_2d();
                let b_pose = sf::Pose::new(
                    player_pos + state.facing.orient_vec(sf::Vec2::new(0.2, 0.)),
                    sf::Angle::default(),
                )
                .with_depth(pose.translation.z);
                let b_coll = sf::Collider::new_circle(R);
                let b_body = sf::Body::new_dynamic_const_mass(b_coll.info(), 1.0).with_velocity(
                    sf::Velocity {
                        angular: 0.0,
                        linear: state.facing.orient_vec(sf::Vec2::new(20.0, 0.1)).conv_p(),
                    },
                );
                let b_body_key = game.physics.entity_set.insert_body(b_body);
                let b_coll_key = game.physics.entity_set.attach_collider(b_body_key, b_coll);

                bullet_to_spawn = Some((
                    player_entity,
                    (b_pose, meshes.bullet, b_body_key, b_coll_key),
                ));
            }
        }

        for bullet in bullet_delete_queue {
            game.world.despawn(bullet).ok();
        }
        if let Some((player_ent, bullet)) = bullet_to_spawn {
            let bullet_ent = game.world.spawn(bullet);
            let bullet_coll = *game
                .world
                .query_one_mut::<&sf::ColliderKey>(bullet_ent)
                .unwrap();
            let player_state = game
                .world
                .query_one_mut::<&mut PlayerState>(player_ent)
                .unwrap();
            player_state.active_bullets.push((bullet_ent, bullet_coll));
        }
    }
}
