use crate::math as m;

/// A (possibly) position-dependent force that is typically
/// fed to a physics solver and applied to all rigid bodies each frame.
pub trait ForceField: Send + Sync {
    fn value_at(&self, position: m::Vec2) -> m::Vec2;
}

#[derive(Clone, Copy, Debug)]
pub struct NoneField;
impl ForceField for NoneField {
    fn value_at(&self, _: m::Vec2) -> m::Vec2 {
        m::Vec2::zero()
    }
}

/// A combination of two different force fields.
#[derive(Clone, Copy, Debug)]
pub struct Sum<F1: ForceField, F2: ForceField>(pub F1, pub F2);
impl<F1: ForceField, F2: ForceField> ForceField for Sum<F1, F2> {
    fn value_at(&self, pos: m::Vec2) -> m::Vec2 {
        self.0.value_at(pos) + self.1.value_at(pos)
    }
}

/// Constant gravity field over all of space.
#[derive(Clone, Copy, Debug)]
pub struct Gravity(pub m::Vec2);
impl ForceField for Gravity {
    fn value_at(&self, _pos: m::Vec2) -> m::Vec2 {
        self.0
    }
}

/// Gravity that pulls towards a specific point in space.
///
/// With a negative `strength` value this can also be a repulsive force.
#[derive(Clone, Copy, Debug)]
pub struct PointGravity {
    /// The position of the gravity source.
    pub position: m::Vec2,
    /// The strength of gravity at the source.
    pub strength: f64,
    /// How quickly gravity falls off with distance.
    pub falloff: f64,
}
impl ForceField for PointGravity {
    fn value_at(&self, pos: m::Vec2) -> m::Vec2 {
        let dist = self.position - pos;
        // + 1.0 so that the divisor is 1 at the source
        let strength = self.strength / ((dist.mag_sq() + 1.0) * self.falloff);
        strength * dist.normalized()
    }
}
