mod animation;
pub use animation::animator::Animator;

mod manager;
pub use manager::{AnimationId, GraphicsManager, MaterialId, MeshId};

mod scene;
pub use scene::Scene;

pub mod renderer;
pub use renderer::Renderer;

pub(crate) mod gi;

pub mod util;

pub mod camera;
pub use camera::{Camera, MouseDragCameraController};

pub(super) mod mesh;
pub use mesh::{ConvexMeshShape, Mesh, MeshRenderer, Skin, Vertex as MeshVertex};

mod line_renderer;
pub use line_renderer::{LineStrip, LineVertex};

pub mod material;
pub use material::Texture;
