use std::collections::HashMap;
use thunderdome as td;

use super::{
    material::{Material, MaterialDefaults},
    Mesh, Renderer, Skin,
};

#[cfg(feature = "gltf")]
pub(crate) mod gltf_import;

/// Key type to look up a mesh loaded into the [`GraphicsManager`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MeshKey(td::Index);

pub struct GraphicsManager {
    meshes: td::Arena<Mesh>,
    mesh_id_map: HashMap<String, MeshKey>,
    materials: td::Arena<Material>,
    mat_defaults: MaterialDefaults,
    skins: td::Arena<Skin>,
    // TODO: animations
}

/// Error when loading assets from a glTF document.
#[cfg(feature = "gltf")]
#[derive(thiserror::Error, Debug)]
pub enum LoadError {
    #[error("Invalid file name")]
    InvalidFileName,
    #[error("Failed to open the file at the given path")]
    IoError(#[from] std::io::Error),
    #[error("Failed to read glTF document")]
    GltfError(#[from] gltf::Error),
}

impl GraphicsManager {
    /// Create a new graphics manager.
    #[inline]
    pub fn new(rend: &Renderer) -> Self {
        Self {
            meshes: td::Arena::new(),
            mesh_id_map: HashMap::new(),
            materials: td::Arena::new(),
            mat_defaults: MaterialDefaults::new(rend),
            skins: td::Arena::new(),
        }
    }

    /// Load all graphics assets (meshes, skins, materials, animations)
    /// in a glTF document.
    #[cfg(feature = "gltf")]
    pub fn load_gltf(
        &mut self,
        rend: &Renderer,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), LoadError> {
        let path = path.as_ref();
        let file_bytes = std::fs::read(path)?;

        let file_stem = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or(LoadError::InvalidFileName)?;
        // asset ids are names of the gltf node prefixed with file name
        let name_to_id = |name: &str| format!("{file_stem}.{name}");

        let (doc, bufs, images) = gltf::import_slice(file_bytes)?;
        let bufs: Vec<&[u8]> = bufs.iter().map(|data| data.0.as_slice()).collect();

        for mut mesh in gltf_import::load_meshes(&doc, &bufs, name_to_id) {
            mesh.upload(&rend.device);
            self.insert_mesh(mesh);
        }

        for mat_params in gltf_import::load_materials(&doc, &images) {
            self.materials
                .insert(Material::new(rend, &self.mat_defaults, mat_params));
        }

        for skin in gltf_import::load_skins(&doc, &bufs) {
            self.skins.insert(skin);
        }

        Ok(())
    }

    /// Add a mesh to the set of drawable assets.
    ///
    /// Returns a key that can be used to render the mesh
    /// by inserting it into a [`hecs`] world.
    /// If the mesh has an `id`, this key can also be looked up with
    /// [`get_mesh_key`][Self::get_mesh_key].
    pub fn insert_mesh(&mut self, mesh: Mesh) -> MeshKey {
        let id = mesh.id.clone();
        let key = MeshKey(self.meshes.insert(mesh));
        if let Some(id) = id {
            self.mesh_id_map.insert(id, key);
        }
        key
    }

    /// Get the compact [`MeshKey`] associated with a string id.
    #[inline]
    pub fn get_mesh_key(&self, id: &str) -> Option<MeshKey> {
        self.mesh_id_map.get(id).cloned()
    }

    /// Get a mesh by its [`MeshKey`].
    ///
    /// Currently panics if the mesh is not in the collection anymore
    /// (although there's no API to remove things yet so this should not happen).
    /// TODO: return a default mesh instead
    #[inline]
    pub fn get_mesh(&self, key: MeshKey) -> &Mesh {
        self.meshes.get(key.0).expect("Mesh not found")
    }
}
