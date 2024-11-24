use super::Velocity;
use crate::math::PhysicsPose;

/// A body is something that moves, typically a physics-enabled rigid body or particle.
/// Connect a Body with a Collider to make it collide with other things.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    pub pose: PhysicsPose,
    pub velocity: Velocity,
    pub mass: Mass,
    pub moment_of_inertia: Mass,
    pub ignores_gravity: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct ColliderInfo {
    pub area: f64,
    pub second_moment_of_area: f64,
}

impl Body {
    /// A particle responds to external forces but does not rotate.
    pub fn new_particle(mass: f64) -> Self {
        Self {
            pose: PhysicsPose::default(),
            velocity: Velocity::default(),
            mass: Mass::from(mass),
            moment_of_inertia: Mass::Infinite,
            ignores_gravity: false,
        }
    }

    /// Dynamic bodies respond to external forces and are allowed to rotate.
    /// This constructor calculates mass and moment of inertia from the given density and
    /// collider shape.
    pub fn new_dynamic(coll_info: ColliderInfo, density: f64) -> Self {
        let mass = coll_info.area * density;
        Self {
            pose: PhysicsPose::default(),
            velocity: Velocity::default(),
            mass: Mass::from(mass),
            moment_of_inertia: Mass::from(coll_info.second_moment_of_area * density),
            ignores_gravity: false,
        }
    }

    /// Create a dynamic body with the given mass instead of using density.
    /// The collider is still required in order to compute moment of inertia.
    pub fn new_dynamic_const_mass(coll_info: ColliderInfo, mass: f64) -> Self {
        let density = mass / coll_info.area;
        Self {
            pose: PhysicsPose::default(),
            velocity: Velocity::default(),
            mass: Mass::from(mass),
            moment_of_inertia: Mass::from(coll_info.second_moment_of_area * density),
            ignores_gravity: false,
        }
    }

    /// Kinematic bodies are not affected by collision forces.
    pub fn new_kinematic() -> Self {
        Self {
            pose: PhysicsPose::default(),
            velocity: Velocity::default(),
            mass: Mass::Infinite,
            moment_of_inertia: Mass::Infinite,
            ignores_gravity: false,
        }
    }

    /// Set the pose of the body in a builder-like chain.
    pub fn with_pose(mut self, pose: PhysicsPose) -> Self {
        self.pose = pose;
        self
    }

    /// Set the velocity of the body in a builder-like chain.
    pub fn with_velocity(mut self, vel: Velocity) -> Self {
        self.velocity = vel;
        self
    }

    /// Stop this body from being accelerated by gravity.
    pub fn ignore_gravity(mut self) -> Self {
        self.ignores_gravity = true;
        self
    }

    /// Check whether the body has finite mass or moment of inertia, allowing forces to have an
    /// effect on it.
    #[inline]
    pub fn sees_forces(&self) -> bool {
        !matches!(
            (self.mass, self.moment_of_inertia),
            (Mass::Infinite, Mass::Infinite)
        )
    }
}

/// Mass or moment of inertia of a body, which can be infinite.
///
/// This stores both a mass value and its inverse, because calculating inverse mass
/// is expensive and needed a lot in physics calculations.
#[derive(Clone, Copy, Debug)]
pub enum Mass {
    Finite { mass: f64, inverse: f64 },
    Infinite,
}

impl From<f64> for Mass {
    #[inline]
    fn from(mass: f64) -> Self {
        Mass::Finite {
            mass,
            inverse: 1.0 / mass,
        }
    }
}

impl Mass {
    /// Get the inverse of the mass, which is zero if the mass is infinite.
    #[inline]
    pub fn inv(&self) -> f64 {
        match self {
            Mass::Finite { inverse, .. } => *inverse,
            Mass::Infinite => 0.0,
        }
    }
}
