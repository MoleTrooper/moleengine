// normally I wouldn't use a glob import but this module uses like actually everything from super
use super::*;

#[derive(Clone, Copy, Debug)]
pub struct RopeView {
    pub params: rope::RopeParameters,
    pub start: usize,
}

/// View into the working buffers created in physics::tick
/// and some other data for a single island.
pub struct DataView<'a> {
    pub dt: f64,
    pub inv_dt: f64,
    pub inv_dt_sq: f64,
    /// index of the first body in the island in the global buffers
    pub island_offset: usize,
    /// map from the entity_set body storage to the sorted order
    pub global_body_order: &'a [usize],
    /// slice of sorted bodies that are part of the island
    pub bodies: &'a mut [Body],
    pub old_poses: &'a mut [PhysicsPose],
    pub pre_contact_poses: &'a mut [PhysicsPose],
    pub old_velocities: &'a mut [Velocity],
    pub ext_f_accelerations: &'a mut [uv::DVec2],
    pub ropes: &'a mut [RopeView],
    pub rope_next_particles: &'a [Option<usize>],
    pub rope_prev_particles: &'a [Option<usize>],
    pub rope_lateral_corrections: &'a mut [Option<uv::DVec2>],
    pub constraints: &'a [Constraint],
    pub constraint_body_pairs: &'a [(usize, Option<usize>)],
    pub coll_pairs: &'a [[ColliderKey; 2]],
    pub contacts: &'a mut [ContactResult],
    pub last_contacts: &'a mut [ContactResult],
    pub contact_lambdas: &'a mut [f64],
}

// SAFETY: we only use these inside of the solver
// where we make sure we don't drop anything before every view is solved
unsafe impl<'a> Sync for DataView<'a> {}
unsafe impl<'a> Send for DataView<'a> {}

/// Get the index of the body connected to a collider within this island's slice.
fn get_collider_body(
    global_body_order: &[usize],
    island_offset: usize,
    coll_key: ColliderKey,
    entity_set: &EntitySet,
) -> Option<usize> {
    let body_key = entity_set.coll_bodies.get(coll_key.0)?;
    let slot = body_key.0.slot() as usize;
    Some(global_body_order[slot] - island_offset)
}

pub fn solve(forcefield: &impl ForceField, data: &mut DataView<'_>, entity_set: &EntitySet) {
    // apply external forces and estimate post-step pose with explicit Euler step
    for (body, old_pose, old_vel, ext_accel) in izip!(
        &mut *data.bodies,
        &mut *data.old_poses,
        &mut *data.old_velocities,
        &mut *data.ext_f_accelerations
    ) {
        if let Mass::Finite { .. } = body.mass {
            // TODO: rename forcefield to accelerationfield or allow it to depend on mass
            let ff_accel = forcefield.value_at(body.pose.translation);
            body.velocity.linear += ff_accel * data.dt;
            *ext_accel = ff_accel;
        }

        // old_vel is velocity after external forces but before collisions
        *old_vel = body.velocity;
        *old_pose = body.pose;
        body.pose = body.velocity.apply_to_pose(data.dt, body.pose);
    }

    if !data.ropes.is_empty() {
        solve_ropes(data);
    }
    if !data.constraints.is_empty() {
        solve_constraints(data);
    }

    for (body, pre_cont_pose) in izip!(&mut *data.bodies, &mut *data.pre_contact_poses) {
        *pre_cont_pose = body.pose;
    }
    if !data.contacts.is_empty() {
        solve_contacts(data, entity_set);
    }

    // update velocities from pose differences
    for (old_pose, body) in izip!(&mut *data.old_poses, &mut *data.bodies,) {
        body.velocity.linear = (body.pose.translation - old_pose.translation) * data.inv_dt;
        // I'm sure there are more efficient ways to handle the angle but this'll do
        let pose_diff = body.pose.rotation * old_pose.rotation.reversed();
        let angle_diff = -pose_diff.bv.xy.atan2(pose_diff.s) * 2.0;
        body.velocity.angular = angle_diff * data.inv_dt;
    }

    if !data.contacts.is_empty() {
        contact_velocity_step(data, entity_set);
    }
    if !data.constraints.is_empty() {
        constraint_damping(data);
    }
    if !data.ropes.is_empty() {
        rope_velocity_step(data);
    }
}

//
// Solve ropes
//

fn solve_ropes(data: &mut DataView<'_>) {
    let _span = tracy_client::span!("solve ropes");

    for rope in &*data.ropes {
        let mut curr_particle = rope.start;
        let mut next_particle =
            data.rope_next_particles[curr_particle].expect("Rope only had one particle");
        loop {
            let dist = data.bodies[next_particle].pose.translation
                - data.bodies[curr_particle].pose.translation;
            let dist_mag = dist.mag();
            let dir = dist / dist_mag;
            let error = rope.params.spacing - dist_mag;

            let lambda = -error
                / (data.bodies[curr_particle].mass.inv()
                    + data.bodies[next_particle].mass.inv()
                    + rope.params.compliance * data.inv_dt_sq);

            data.bodies[curr_particle]
                .pose
                .append_translation(data.bodies[curr_particle].mass.inv() * lambda * dir);
            data.bodies[next_particle]
                .pose
                .append_translation(-data.bodies[next_particle].mass.inv() * lambda * dir);

            let particle_after_next = match data.rope_next_particles[next_particle] {
                Some(next) => next,
                None => break,
            };

            // curvature constraint between last three particles

            let curr_to_next = data.bodies[next_particle].pose.translation
                - data.bodies[curr_particle].pose.translation;
            let next_to_after = data.bodies[particle_after_next].pose.translation
                - data.bodies[next_particle].pose.translation;
            let angle = next_to_after
                .normalized()
                .dot(curr_to_next.normalized())
                .acos();
            let error = angle - rope.params.bending_max_angle;
            if error > 0.0 {
                let lambda = -error
                    / (data.bodies[particle_after_next].mass.inv()
                        + rope.params.bending_compliance * data.inv_dt_sq);

                let lambda_oriented = if left_normal(curr_to_next).dot(next_to_after) > 0.0 {
                    lambda
                } else {
                    -lambda
                };
                let correction = uv::DRotor2::from_angle(
                    lambda_oriented * data.bodies[particle_after_next].mass.inv(),
                );
                let old_pos = data.bodies[particle_after_next].pose.translation;
                data.bodies[particle_after_next].pose.translation =
                    data.bodies[next_particle].pose.translation + correction * next_to_after;

                data.rope_lateral_corrections[particle_after_next] =
                    Some(data.bodies[particle_after_next].pose.translation - old_pos);
            }

            curr_particle = next_particle;
            next_particle = particle_after_next;
        }
    }
}

//
// Solve constraints
//

fn solve_constraints(data: &mut DataView<'_>) {
    let _span = tracy_client::span!("solve constraints");

    for (constraint, pair) in izip!(data.constraints, data.constraint_body_pairs) {
        let inv_masses = map_semi_pair(*pair, |b| data.bodies[*b].mass.inv(), 0.0);
        let inv_mom_inertias =
            map_semi_pair(*pair, |b| data.bodies[*b].moment_of_inertia.inv(), 0.0);

        match constraint.ty {
            ConstraintType::Distance { distance } => {
                let offsets_worldspace = [
                    data.bodies[pair.0].pose * constraint.offsets[0],
                    pair.1
                        .map(|p1| data.bodies[p1].pose * constraint.offsets[1])
                        .unwrap_or(constraint.offsets[1]),
                ];
                let actual_dist = offsets_worldspace[1] - offsets_worldspace[0];
                let actual_dist_mag = actual_dist.mag();
                let error = distance - actual_dist_mag;

                if match constraint.limit {
                    ConstraintLimit::Eq => true,
                    ConstraintLimit::Lt if error < 0.0 => true,
                    ConstraintLimit::Gt if error > 0.0 => true,
                    _ => false,
                } {
                    let dir = if actual_dist_mag != 0.0 {
                        actual_dist / actual_dist_mag
                    } else {
                        uv::DVec2::unit_y()
                    };

                    match pair.1 {
                        Some(p1) => {
                            let pair = [pair.0, p1];
                            let offsets_rotated = map_pair(&[0, 1], |&i| {
                                data.bodies[pair[i]].pose.rotation * constraint.offsets[i]
                            });
                            let offsets_wedge_dir =
                                map_pair(&[0, 1], |&i| offsets_rotated[i].wedge(dir).xy);
                            let eff_inv_masses = map_pair(&[0, 1], |&i| {
                                inv_masses[i] + (offsets_wedge_dir[i].powi(2) * inv_mom_inertias[i])
                            });

                            let lambda = -error
                                / (eff_inv_masses[0]
                                    + eff_inv_masses[1]
                                    + constraint.compliance * data.inv_dt_sq);

                            let p0 = &mut data.bodies[pair[0]].pose;
                            p0.append_translation(inv_masses[0] * lambda * dir);
                            p0.prepend_rotation(uv::DRotor2::from_angle(
                                inv_mom_inertias[0] * lambda * offsets_wedge_dir[0],
                            ));
                            let p1 = &mut data.bodies[pair[1]].pose;
                            p1.append_translation(-inv_masses[1] * lambda * dir);
                            p1.prepend_rotation(uv::DRotor2::from_angle(
                                -inv_mom_inertias[1] * lambda * offsets_wedge_dir[1],
                            ));
                        }
                        None => {
                            // this is repetitive but kind of hard to abstract :thinking:
                            let offset_rotated =
                                data.bodies[pair.0].pose.rotation * constraint.offsets[0];
                            let offset_wedge_dir = offset_rotated.wedge(dir).xy;
                            let eff_inv_mass =
                                inv_masses[0] + offset_wedge_dir.powi(2) * inv_mom_inertias[0];

                            let lambda =
                                -error / (eff_inv_mass + constraint.compliance * data.inv_dt_sq);

                            let p0 = &mut data.bodies[pair.0].pose;
                            p0.append_translation(inv_masses[0] * lambda * dir);
                            p0.prepend_rotation(uv::DRotor2::from_angle(
                                inv_mom_inertias[0] * lambda * offset_wedge_dir,
                            ));
                        }
                    }
                }
            }
        }
    }
}

//
// Solve contacts
//

fn solve_contacts(data: &mut DataView<'_>, entity_set: &EntitySet) {
    let _span = tracy_client::span!("solve contacts");

    for (coll_keys, contact, last_contact, lambda_n) in izip!(
        data.coll_pairs,
        &mut *data.contacts,
        &mut *data.last_contacts,
        &mut *data.contact_lambdas
    ) {
        let bodies: [Option<usize>; 2] = map_pair(coll_keys, |c| {
            get_collider_body(data.global_body_order, data.island_offset, *c, entity_set)
        });

        if bodies[0].is_none() && bodies[1].is_none() {
            // both colliders are static, skip this pair.
            // for kinematic bodies we still report contacts but skip physics
            *contact = ContactResult::Zero;
            continue;
        }

        let colls: [&Collider; 2] = map_pair(coll_keys, |c| entity_set.colliders.get(c.0).unwrap());
        // if a body is attached,
        // the pose of a collider is in the local space of the body
        let poses_worldspace: [PhysicsPose; 2] = map_pair(&[0, 1], |&i| match bodies[i] {
            Some(bi) => data.bodies[bi].pose * colls[i].pose,
            None => colls[i].pose,
        });

        // check for collision
        *contact = collision::shape_shape::intersection_check(
            poses_worldspace,
            [colls[0].shape, colls[1].shape],
        )
        .map(|mut cont| {
            // transform contact to local space of bodies if attached
            cont.offsets = map_pair(&[0, 1], |&i| match bodies[i] {
                Some(_) => colls[i].pose * cont.offsets[i],
                None => cont.offsets[i],
            });
            cont
        });
        // mark latest contact that wasn't zero;
        // this will be available to be queried by the user later
        if !matches!(contact, ContactResult::Zero) {
            *last_contact = *contact;
        }

        // if both bodies are static or kinematic, stop here
        if !bodies[0]
            .map(|bi| data.bodies[bi].sees_forces())
            .unwrap_or(false)
            && !bodies[1]
                .map(|bi| data.bodies[bi].sees_forces())
                .unwrap_or(false)
        {
            continue;
        }

        // if one of the bodies is from a rope, adjust normal
        // to perpendicular to the rope *before* any contacts
        //
        // (because rope colliders are circles, only the One case is possible here)
        if let ContactResult::One(contact) = contact {
            for (body, normal_dir) in izip!(bodies, [1.0, -1.0]) {
                if let Some(bi) = body {
                    if let (Some(prev), Some(next)) =
                        (data.rope_prev_particles[bi], data.rope_next_particles[bi])
                    {
                        let normal_oriented = *contact.normal * normal_dir;
                        let to_prev = data.pre_contact_poses[prev].translation
                            - data.pre_contact_poses[bi].translation;
                        let to_next = data.pre_contact_poses[next].translation
                            - data.pre_contact_poses[bi].translation;
                        let new_normal =
                            if normal_oriented.dot(to_prev) > normal_oriented.dot(to_next) {
                                UnitDVec2::new_normalize(left_normal(to_prev))
                            } else {
                                UnitDVec2::new_normalize(left_normal(to_next))
                            };
                        contact.normal = if contact.normal.dot(*new_normal) > 0.0 {
                            new_normal
                        } else {
                            -new_normal
                        };
                    }
                }
            }
        }

        let materials = match (colls[0].ty, colls[1].ty) {
            (ColliderType::Solid(m0), ColliderType::Solid(m1)) => [m0, m1],
            // one of the colliders was a trigger, no physics response
            _ => {
                continue;
            }
        };

        for contact in contact.iter() {
            // tangent for static friction
            let tangent = left_normal(*contact.normal);

            // gather variables into a struct because they're different
            // for static and dynamic bodies and this lets us get them in one match
            struct WorkingVars {
                // we can't return depth directly from collision detection because
                // earlier position corrections can change it,
                // thus we compute depth here from the points on each object's surface
                offset_worldspace: uv::DVec2,
                offset_wedge_normal: f64,
                eff_inv_mass_n: f64,
                // for friction
                offset_worldspace_old: uv::DVec2,
                offset_wedge_tan: f64,
                eff_inv_mass_tan: f64,
            }
            let vars = map_pair(&[0, 1], |&i| {
                match bodies[i] {
                    // no body attached -> static body, infinite mass
                    None => {
                        let offset_worldspace = colls[i].pose * contact.offsets[i];
                        WorkingVars {
                            offset_worldspace,
                            offset_wedge_normal: 0.0,
                            eff_inv_mass_n: 0.0,
                            offset_worldspace_old: offset_worldspace,
                            offset_wedge_tan: 0.0,
                            eff_inv_mass_tan: 0.0,
                        }
                    }
                    Some(bi) => {
                        let im = data.bodies[bi].mass.inv();
                        let imi = data.bodies[bi].moment_of_inertia.inv();
                        // do NOT apply collider pose here;
                        // contact was transformed into the body's local space
                        let offset_rotated = data.bodies[bi].pose.rotation * contact.offsets[i];
                        let offset_wedge_normal = offset_rotated.wedge(*contact.normal).xy;
                        let offset_wedge_tan = offset_rotated.wedge(tangent).xy;

                        WorkingVars {
                            offset_worldspace: data.bodies[bi].pose * contact.offsets[i],
                            offset_wedge_normal,
                            eff_inv_mass_n: im + (offset_wedge_normal.powi(2) * imi),
                            offset_worldspace_old: data.old_poses[bi] * contact.offsets[i],
                            offset_wedge_tan,
                            eff_inv_mass_tan: im + (offset_wedge_tan.powi(2) * imi),
                        }
                    }
                }
            });

            let depth =
                (vars[0].offset_worldspace - vars[1].offset_worldspace).dot(*contact.normal);

            if depth <= 0.0 {
                *lambda_n = 0.0;
                continue;
            }

            *lambda_n = -depth / (vars[0].eff_inv_mass_n + vars[1].eff_inv_mass_n);

            if let Some(bi) = bodies[0] {
                let im = data.bodies[bi].mass.inv();
                let imi = data.bodies[bi].moment_of_inertia.inv();
                let p = &mut data.bodies[bi].pose;
                p.append_translation(im * *lambda_n * *contact.normal);
                p.prepend_rotation(uv::DRotor2::from_angle(
                    imi * *lambda_n * vars[0].offset_wedge_normal,
                ));
            }
            if let Some(bi) = bodies[1] {
                let im = data.bodies[bi].mass.inv();
                let imi = data.bodies[bi].moment_of_inertia.inv();
                let p = &mut data.bodies[bi].pose;
                p.append_translation(-im * *lambda_n * *contact.normal);
                p.prepend_rotation(uv::DRotor2::from_angle(
                    -imi * *lambda_n * vars[1].offset_wedge_normal,
                ));
            }

            // static friction

            if let Some(friction_coef) = materials[0].static_friction_with(&materials[1]) {
                let offset_diff_motion = (vars[0].offset_worldspace
                    - vars[0].offset_worldspace_old)
                    - (vars[1].offset_worldspace - vars[1].offset_worldspace_old);
                let motion_along_tan = offset_diff_motion.dot(tangent);

                let max_coulomb_dx = *lambda_n * friction_coef;

                let lambda_t =
                    -motion_along_tan / (vars[0].eff_inv_mass_tan + vars[1].eff_inv_mass_tan);

                if lambda_t < max_coulomb_dx {
                    if let Some(bi) = bodies[0] {
                        let im = data.bodies[bi].mass.inv();
                        let imi = data.bodies[bi].moment_of_inertia.inv();
                        let p = &mut data.bodies[bi].pose;
                        p.append_translation(im * lambda_t * tangent);
                        p.prepend_rotation(uv::DRotor2::from_angle(
                            imi * lambda_t * vars[0].offset_wedge_tan,
                        ));
                    }
                    if let Some(bi) = bodies[1] {
                        let im = data.bodies[bi].mass.inv();
                        let imi = data.bodies[bi].moment_of_inertia.inv();
                        let p = &mut data.bodies[bi].pose;
                        p.append_translation(-im * lambda_t * tangent);
                        p.prepend_rotation(uv::DRotor2::from_angle(
                            -imi * lambda_t * vars[1].offset_wedge_tan,
                        ));
                    }
                }
            }
        }
    }
}

//
// Contact velocity step
//

fn contact_velocity_step(data: &mut DataView<'_>, entity_set: &EntitySet) {
    let _span = tracy_client::span!("contact velocity step");

    for (coll_keys, contact, lambda_n) in
        izip!(data.coll_pairs, &*data.contacts, &*data.contact_lambdas)
    {
        let colls: [&Collider; 2] = map_pair(coll_keys, |c| entity_set.colliders.get(c.0).unwrap());

        let materials = match (colls[0].ty, colls[1].ty) {
            (ColliderType::Solid(m0), ColliderType::Solid(m1)) => [m0, m1],
            _ => {
                // one of the colliders was a sensor, no physics response
                continue;
            }
        };

        let bodies: [Option<usize>; 2] = map_pair(coll_keys, |c| {
            get_collider_body(data.global_body_order, data.island_offset, *c, entity_set)
        });

        // check if the contact is between two kinematic/static bodies, skip if so
        if !bodies[0]
            .map(|bi| data.bodies[bi].sees_forces())
            .unwrap_or(false)
            && !bodies[1]
                .map(|bi| data.bodies[bi].sees_forces())
                .unwrap_or(false)
        {
            continue;
        }

        for contact in contact.iter() {
            struct WorkingVars {
                inv_mass: f64,
                inv_mom_inertia: f64,
                offset_rotated: uv::DVec2,
                point_vel: uv::DVec2,
                old_point_vel: uv::DVec2,
                ext_f_accel: uv::DVec2,
            }
            let vars = map_pair(&[0, 1], |&i| match bodies[i] {
                // no body => infinite mass
                None => WorkingVars {
                    inv_mass: 0.0,
                    inv_mom_inertia: 0.0,
                    offset_rotated: colls[i].pose.rotation * contact.offsets[i],
                    point_vel: uv::DVec2::zero(),
                    old_point_vel: uv::DVec2::zero(),
                    ext_f_accel: uv::DVec2::zero(),
                },
                Some(bi) => {
                    // the contact was transformed into the body's local space in `solve_contacts`,
                    // thus no collider pose applied here
                    let offset_rotated = data.bodies[bi].pose.rotation * contact.offsets[i];
                    WorkingVars {
                        inv_mass: data.bodies[bi].mass.inv(),
                        inv_mom_inertia: data.bodies[bi].moment_of_inertia.inv(),
                        offset_rotated,
                        point_vel: data.bodies[bi].velocity.point_velocity(offset_rotated),
                        old_point_vel: data.old_velocities[bi].point_velocity(offset_rotated),
                        ext_f_accel: data.ext_f_accelerations[bi],
                    }
                }
            });

            let relative_vel_at_p = vars[0].point_vel - vars[1].point_vel;

            // restitution

            let normal_vel = relative_vel_at_p.dot(*contact.normal);
            let old_rel_vel = vars[0].old_point_vel - vars[1].old_point_vel;
            let old_normal_vel = old_rel_vel.dot(*contact.normal);
            let restitution_coef = if old_normal_vel * old_normal_vel
                < data.dt * data.dt * (vars[0].ext_f_accel + vars[1].ext_f_accel).mag_sq()
            {
                // don't bounce if the normal velocity is very small to avoid jitter
                0.0
            } else {
                materials[0].restitution_with(&materials[1])
            };
            let delta_normal_vel = -normal_vel - restitution_coef * old_normal_vel.max(0.0);

            // dynamic friction

            let tangent = left_normal(*contact.normal);
            let delta_tan_vel = match materials[0].dynamic_friction_with(&materials[1]) {
                Some(friction_coef) => {
                    let tangent_vel = relative_vel_at_p.dot(tangent);
                    let max_coulomb_dv = data.inv_dt * *lambda_n * friction_coef;
                    tangent_vel.abs().min(max_coulomb_dv.abs()) * -tangent_vel.signum()
                }
                None => 0.0,
            };

            // apply impulse

            let total_vel_update = delta_normal_vel * *contact.normal + delta_tan_vel * tangent;
            let vel_update_mag = total_vel_update.mag();
            if vel_update_mag < 0.0001 {
                continue;
            }
            let vel_update_dir = total_vel_update / vel_update_mag;
            let offsets_wedge_dv = map_pair(&[0, 1], |&i| {
                vars[i].offset_rotated.wedge(vel_update_dir).xy
            });
            let eff_inv_masses = map_pair(&[0, 1], |&i| {
                vars[i].inv_mass + (offsets_wedge_dv[i].powi(2) * vars[i].inv_mom_inertia)
            });
            let impulse_mag = vel_update_mag / (eff_inv_masses[0] + eff_inv_masses[1]);

            if let Some(bi) = bodies[0] {
                data.bodies[bi].velocity.linear += vars[0].inv_mass * impulse_mag * vel_update_dir;
                data.bodies[bi].velocity.angular +=
                    vars[0].inv_mom_inertia * impulse_mag * offsets_wedge_dv[0];
            }
            if let Some(bi) = bodies[1] {
                data.bodies[bi].velocity.linear -= vars[1].inv_mass * impulse_mag * vel_update_dir;
                data.bodies[bi].velocity.angular -=
                    vars[1].inv_mom_inertia * impulse_mag * offsets_wedge_dv[1];
            }
        }
    }
}

//
// Constraint damping
//

fn constraint_damping(data: &mut DataView<'_>) {
    let _span = tracy_client::span!("constraint damping");

    for (constraint, pair) in izip!(data.constraints, data.constraint_body_pairs) {
        let inv_masses = map_semi_pair(*pair, |b| data.bodies[*b].mass.inv(), 0.0);
        let inv_mom_inertias =
            map_semi_pair(*pair, |b| data.bodies[*b].moment_of_inertia.inv(), 0.0);

        match pair.1 {
            Some(p1) => {
                let pair = [pair.0, p1];
                let offsets_rotated = map_pair(&[0, 1], |&i| {
                    data.bodies[pair[i]].pose.rotation * constraint.offsets[i]
                });

                let relative_vel = data.bodies[pair[0]]
                    .velocity
                    .point_velocity(offsets_rotated[0])
                    - data.bodies[pair[1]]
                        .velocity
                        .point_velocity(offsets_rotated[1]);
                let relative_vel_mag = relative_vel.mag();
                if relative_vel_mag == 0.0 {
                    continue;
                }
                let dir = relative_vel / relative_vel_mag;

                let offsets_wedge_dir = map_pair(&[0, 1], |&i| offsets_rotated[i].wedge(dir).xy);
                let eff_inv_masses = map_pair(&[0, 1], |&i| {
                    inv_masses[i] + (offsets_wedge_dir[i].powi(2) * inv_mom_inertias[i])
                });

                let vel_update_mag =
                    -relative_vel_mag * (constraint.linear_damping * data.dt).min(1.0);
                let linear_impulse_mag = vel_update_mag / (eff_inv_masses[0] + eff_inv_masses[1]);

                data.bodies[pair[0]].velocity.linear += inv_masses[0] * linear_impulse_mag * dir;
                data.bodies[pair[0]].velocity.angular +=
                    inv_mom_inertias[0] * linear_impulse_mag * offsets_wedge_dir[0];
                data.bodies[pair[1]].velocity.linear -= inv_masses[1] * linear_impulse_mag * dir;
                data.bodies[pair[1]].velocity.angular -=
                    inv_mom_inertias[1] * linear_impulse_mag * offsets_wedge_dir[1];

                if constraint.angular_damping > 0.0 {
                    let rel_angular_vel = data.bodies[pair[0]].velocity.angular
                        - data.bodies[pair[1]].velocity.angular;
                    let ang_vel_update_mag =
                        -rel_angular_vel * (constraint.angular_damping * data.dt).min(1.0);
                    let angular_impulse =
                        ang_vel_update_mag / (eff_inv_masses[0] + eff_inv_masses[1]);

                    data.bodies[pair[1]].velocity.angular -= inv_mom_inertias[1] * angular_impulse;
                    data.bodies[pair[0]].velocity.angular += inv_mom_inertias[0] * angular_impulse;
                };
            }
            None => {
                let offset_rotated = data.bodies[pair.0].pose.rotation * constraint.offsets[0];

                let point_vel = data.bodies[pair.0].velocity.point_velocity(offset_rotated);
                let point_vel_mag = point_vel.mag();
                if point_vel_mag == 0.0 {
                    continue;
                }
                let dir = point_vel / point_vel_mag;

                let offset_wedge_dir = offset_rotated.wedge(dir).xy;
                let eff_inv_mass = inv_masses[0] + offset_wedge_dir.powi(2) * inv_mom_inertias[0];

                let vel_update_mag =
                    -point_vel_mag * (constraint.linear_damping * data.dt).min(1.0);
                let linear_impulse_mag = vel_update_mag / eff_inv_mass;

                data.bodies[pair.0].velocity.linear += inv_masses[0] * linear_impulse_mag * dir;
                data.bodies[pair.0].velocity.angular +=
                    inv_mom_inertias[0] * linear_impulse_mag * offset_wedge_dir;

                if constraint.angular_damping > 0.0 {
                    let ang_vel_update_mag = data.bodies[pair.0].velocity.angular
                        * (constraint.angular_damping * data.dt).min(1.0);
                    let angular_impulse = -ang_vel_update_mag / eff_inv_mass;
                    data.bodies[pair.0].velocity.angular += inv_mom_inertias[0] * angular_impulse;
                };
            }
        }
    }
}

//
// Rope velocity step
//

fn rope_velocity_step(data: &mut DataView<'_>) {
    let _span = tracy_client::span!("rope velocity step");

    for rope in &*data.ropes {
        let mut curr_particle = rope.start;
        let mut next_particle = data.rope_next_particles[curr_particle].unwrap();
        loop {
            let relative_vel = data.bodies[curr_particle].velocity.linear
                - data.bodies[next_particle].velocity.linear;
            let relative_vel_mag = relative_vel.mag();
            if relative_vel_mag != 0.0 {
                let dir = relative_vel / relative_vel_mag;
                let vel_update_mag = -relative_vel_mag * (rope.params.damping * data.dt).min(1.0);

                let linear_impulse_mag = vel_update_mag
                    / (data.bodies[curr_particle].mass.inv()
                        + data.bodies[next_particle].mass.inv());

                data.bodies[curr_particle].velocity.linear +=
                    data.bodies[curr_particle].mass.inv() * linear_impulse_mag * dir;
                data.bodies[next_particle].velocity.linear -=
                    data.bodies[next_particle].mass.inv() * linear_impulse_mag * dir;
            }

            curr_particle = next_particle;
            next_particle = match data.rope_next_particles[next_particle] {
                Some(next) => next,
                None => break,
            };
        }

        // velocity correction to prevent bouncing if there was a lateral position correction
        let mut particle = rope.start;
        loop {
            if let Some(corr) = data.rope_lateral_corrections[particle] {
                let corr_mag = corr.mag();
                // velocity "created" by the correction, used as a maximum bound on
                // velocity correction to keep velocity from e.g. gravity
                let vel_from_corr = corr_mag * data.inv_dt;

                let dir = corr / corr_mag;
                let vel_in_dir = data.bodies[particle].velocity.linear.dot(dir);
                let vel_clamped = vel_in_dir.min(vel_from_corr).max(-vel_from_corr);

                let impulse_mag = -vel_clamped
                    / (data.bodies[particle].mass.inv()
                        + rope.params.bending_compliance * data.inv_dt);
                data.bodies[particle].velocity.linear +=
                    data.bodies[particle].mass.inv() * impulse_mag * dir;
            }

            particle = match data.rope_next_particles[particle] {
                Some(next) => next,
                None => break,
            }
        }
    }
}

#[inline]
fn map_pair<T, R>(pair: &[T; 2], f: impl Fn(&T) -> R) -> [R; 2] {
    [f(&pair[0]), f(&pair[1])]
}

#[inline]
fn map_semi_pair<T, R>(pair: (T, Option<T>), f: impl Fn(&T) -> R, snd_default: R) -> [R; 2] {
    [f(&pair.0), pair.1.map(|x| f(&x)).unwrap_or(snd_default)]
}
