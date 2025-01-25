use crate::math as m;

use super::MeshId;

/// An entity in a scene.
///
/// In glTF, nodes form a tree-shaped hierarchy.
/// Here we flatten the structure such that each entity becomes independent.
/// The hierarchy is retained in skins only,
/// and nonuniform scalings are ignored.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Node {
    pub pose: m::Pose,
    pub mesh: Option<MeshId>,
}

impl Node {
    /// Check that this node has something that interacts with the world
    /// (i.e. it's not just an organizational tree node)
    pub(crate) fn is_valid_entity(&self) -> bool {
        self.mesh.is_some()
    }
}
/// A set of entities to be spawned in the world.
///
/// This format matches the glTF scene format,
/// generally authored in Blender
/// and loaded with [`load_gltf`][crate::GraphicsManager::load_gltf].
/// If not using an external editor,
/// it's probably easier to spawn entities directly in code.
///
/// This is a work in progress, currently only supporting positioning of meshes.
/// More features, such as colliders and custom properties, to come later.
#[derive(Debug, Clone, Default)]
pub struct Scene {
    pub(crate) nodes: Vec<Node>,
}

impl Scene {
    /// Spawn all entities in this scene into the world.
    pub fn spawn(&self, world: &mut hecs::World) {
        for node in self.nodes.iter().filter(|n| n.is_valid_entity()) {
            let ent = world.spawn((node.pose,));
            if let Some(mesh) = node.mesh {
                world.insert_one(ent, mesh).unwrap();
            }
        }
    }
}
