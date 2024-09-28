pub mod game;
pub use game::{Game, GameParams, GameState, GraphicsConfig};

pub mod input;
pub use input::{AxisQuery, Button, ButtonQuery, Input, Key, MouseButton};

pub mod math;
#[cfg(feature = "serde-types")]
pub use math::serde_pose;
pub use math::{uv, Angle, DVec2, PhysicsPose, Pose, PoseBuilder, Rotor2, Rotor3, Vec2, Vec3};

pub mod graphics;
pub use graphics::{
    camera::{Camera, MouseDragCameraController},
    gi::{
        environment_map::{DirectionalLight, EnvironmentMap},
        LightingQualityConfig,
    },
    material::{AttenuationParams, Material, MaterialParams, Texture, TextureData},
    mesh::{ConvexMeshShape, Mesh, MeshData, MeshParams, Skin},
    AnimationId, Animator, GraphicsManager, LineStrip, LineVertex, MaterialId, MeshId, MeshVertex,
    PointLight, Renderer,
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
    hecs_sync::{HecsSyncManager, HecsSyncOptions},
    BodyKey, CastHit, ColliderKey, ConstraintKey, ContactInfo, PhysicsWorld, Rope, RopeKey,
    RopeParameters, RopeSet, Velocity,
};

// re-exported libraries used in public APIs to guarantee versions match
pub use hecs;
pub use wgpu;
pub use winit;
