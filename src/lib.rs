pub mod animation;
pub use animation::{animator, MeshAnimator};

pub mod game;
pub use game::{Game, GameParams, GameState};

pub mod input;
pub use input::{AxisQuery, Button, ButtonQuery, Input, Key, MouseButton};

pub mod math;
#[cfg(feature = "serde-types")]
pub use math::serde_pose;
pub use math::{uv, Angle, Pose, PoseBuilder, Rotor2, Transform, Unit, Vec2};

pub mod graphics;
#[cfg(feature = "gltf")]
pub use graphics::mesh::gltf_import;
pub use graphics::{
    camera::{Camera, CameraScalingStrategy, MouseDragCameraController},
    mesh::{ConvexMeshShape, Mesh, MeshRenderer, Skin},
    DebugVisualizer, OutlineParams, OutlineRenderer, OutlineShape, RenderContext, Renderer,
};

pub mod physics;
pub use physics::{
    body::{Body, ColliderInfo, Mass},
    collision::{
        self, Collider, ColliderPolygon, ColliderShape, ColliderType, CollisionLayerMask,
        CollisionMaskMatrix, CompoundColliderSetup, Contact, ContactResult, PhysicsMaterial, Ray,
        AABB,
    },
    constraint::{Constraint, ConstraintBuilder, ConstraintLimit, ConstraintType},
    forcefield,
    rope::{self, Rope, RopeProperties},
    BodyKey, CastHit, ColliderKey, ConstraintKey, ContactInfo, PhysicsWorld, Velocity,
};

// re-exported libraries used in public APIs to guarantee versions match
pub use hecs;
pub use wgpu;
pub use winit;
