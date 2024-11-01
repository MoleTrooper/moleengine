//! Utilities for loading meshes, skins and animations from glTF documents.

use crate::{
    graphics::{
        animation::gltf_animation as g_anim,
        material::{MaterialParams, TextureData},
        mesh::skin,
        mesh::{self, MeshData},
    },
    math::uv,
};

use itertools::izip;
use std::rc::Rc;

pub fn load_mesh_data<'doc>(buffers: &'doc [&[u8]], prim: gltf::Primitive<'doc>) -> MeshData {
    // helper for constructing gltf readers
    let read_buf = |b: gltf::Buffer| Some(&buffers[b.index()][..b.length()]);

    let reader = prim.reader(read_buf);

    let positions = reader
        .read_positions()
        .expect("Mesh primitive without positions");

    let mut vertices: Vec<mesh::Vertex> = positions
        .into_iter()
        .map(|p| mesh::Vertex {
            // flip z coordinate because starframe uses
            // a left-handed coordinate system (+z away from camera),
            // whereas Blender's is right-handed (+z towards camera)
            position: [p[0], p[1], -p[2]].into(),
            tex_coords: [0., 0.].into(),
            // TODO: also read normals and tangents if available
            ..Default::default()
        })
        .collect();

    if let Some(tex_coords) = reader.read_tex_coords(0) {
        for (vert, uv) in izip!(&mut vertices, tex_coords.into_f32()) {
            vert.tex_coords = uv.into();
        }
    }

    let indices: Vec<u16> = if let Some(idx_read) = reader.read_indices() {
        idx_read
            .into_u32()
            .filter_map(|i| u16::try_from(i).ok())
            .collect()
    } else {
        (0..vertices.len())
            .filter_map(|i| u16::try_from(i).ok())
            .collect()
    };

    let joints = reader.read_joints(0).and_then(|joints| {
        reader.read_weights(0).map(|weights| {
            izip!(joints.into_u16(), weights.into_f32())
                .map(|(j, w)| mesh::VertexJoints {
                    joints: j,
                    weights: w.into(),
                })
                .collect()
        })
    });

    MeshData {
        vertices,
        indices,
        joints,
    }
}

pub fn load_material<'doc>(
    images: &'doc [gltf::image::Data],
    material: gltf::Material<'doc>,
) -> MaterialParams<'doc> {
    let mr = material.pbr_metallic_roughness();

    let base_color = Some(mr.base_color_factor());
    let mat_emissive = material.emissive_factor();
    let emissive_color = if mat_emissive.into_iter().any(|ch| ch != 0.) {
        Some([mat_emissive[0], mat_emissive[1], mat_emissive[2], 1.])
    } else {
        None
    };

    let diffuse_tex = mr.base_color_texture().map(|tex_info| {
        let tex = tex_info.texture();
        let image = &images[tex.source().index()];
        TextureData {
            label: tex.name().map(String::from),
            pixels: &image.pixels,
            format: texture_format_to_wgpu(image.format, true),
            dimensions: (image.width, image.height),
        }
    });

    let normal_tex = material.normal_texture().map(|normal_info| {
        let tex = normal_info.texture();
        let image = &images[tex.source().index()];
        TextureData {
            label: tex.name().map(String::from),
            pixels: &image.pixels,
            format: texture_format_to_wgpu(image.format, false),
            dimensions: (image.width, image.height),
        }
    });

    MaterialParams {
        name: material.name(),
        base_color,
        emissive_color,
        // TODO: figure out where to put attenuation properties of the material.
        // the gltf volume extension is cool
        // but I need this to be a texture which it doesn't have
        attenuation: None,
        diffuse_tex,
        normal_tex,
    }
}

/// Convert a gltf texture format to the wgpu equivalent.
///
/// Gltf doesn't tell you if the textures are SRGB or linear as far as I can tell,
/// so we assume SRGB for diffuse textures and linear for normal maps.
fn texture_format_to_wgpu(format: gltf::image::Format, is_srgb: bool) -> wgpu::TextureFormat {
    use gltf::image::Format as GF;
    use wgpu::TextureFormat as WF;
    match format {
        GF::R8 => WF::R8Unorm,
        GF::R8G8 => WF::Rg8Unorm,
        GF::R8G8B8 => unimplemented!("RGB textures without alpha are not supported"),
        GF::R8G8B8A8 => {
            if is_srgb {
                WF::Rgba8UnormSrgb
            } else {
                WF::Rgba8Unorm
            }
        }
        GF::R16 => WF::R16Unorm,
        GF::R16G16 => WF::Rg16Unorm,
        GF::R16G16B16 => unimplemented!("RGB textures without alpha are not supported"),
        GF::R16G16B16A16 => WF::Rgba16Unorm,
        GF::R32G32B32FLOAT => unimplemented!("RGB textures without alpha are not supported"),
        GF::R32G32B32A32FLOAT => WF::Rgba32Float,
    }
}

/// Load skins from a glTF document.
pub fn load_skin<'doc>(
    buffers: &'doc [&[u8]],
    gltf_skin: gltf::Skin<'doc>,
    // root transform needed
    // because the inverse bind matrices in glTF are relative to the scene root,
    // and we want them relative to the skin root
    root_transform: uv::Mat4,
) -> skin::Skin {
    let read_buf = |b: gltf::Buffer| Some(&buffers[b.index()][..b.length()]);

    let mut joints: Vec<skin::Joint> = gltf_skin
        .joints()
        .map(|joint| skin::Joint {
            name: joint.name().map(String::from),
            // parents will be computed once we have all joints
            parent_idx: None,
            local_pose: skin::TransformDecomp::from_parts(joint.transform().decomposed()),
            // this will also be evaluated later
            global_pose: uv::Mat4::identity(),
        })
        .collect();

    // joint parents

    for (parent_idx, joint) in gltf_skin.joints().enumerate() {
        for child in joint.children() {
            let child_gltf_id = child.index();
            if let Some((child_joint_idx, _)) = gltf_skin
                .joints()
                .enumerate()
                .find(|(_, joint)| joint.index() == child_gltf_id)
            {
                joints[child_joint_idx].parent_idx = Some(parent_idx);
            }
        }
    }

    // inverse bind matrices

    let reader = gltf_skin.reader(read_buf);
    let inv_bind_matrices = if let Some(invs) = reader.read_inverse_bind_matrices() {
        invs.map(uv::Mat4::from).collect()
    } else {
        // inverse bind matrices are not provided, meaning they are premultiplied into vertices
        // (rare case that we can trivially support by making the inverse bind matrices identity,
        // which isn't as efficient as it could be but not worth putting extra effort into)
        vec![uv::Mat4::identity(); joints.len()]
    };

    skin::Skin {
        root_transform,
        joint_set: skin::JointSet { joints },
        inv_bind_matrices: Rc::from(inv_bind_matrices),
        compute_res: None,
    }
}

pub fn load_animation<'doc>(
    buffers: &'doc [&[u8]],
    // animations are assumed to be associated with a single skin
    assoc_skin: gltf::Skin<'doc>,
    gltf_anim: gltf::Animation<'doc>,
) -> g_anim::GltfAnimation {
    let read_buf = |b: gltf::Buffer| Some(&buffers[b.index()][..b.length()]);

    let channels = gltf_anim
        .channels()
        .filter_map(|gltf_chan| {
            use gltf::animation::util::ReadOutputs as Out;
            use gltf::animation::Interpolation as Interp;
            use gltf::animation::Property as Prop;

            let target = gltf_chan.target();
            let target_joint = match assoc_skin
                .joints()
                .enumerate()
                .find(|(_, joint)| joint.index() == target.node().index())
            {
                Some((joint_idx, _)) => joint_idx,
                None => {
                    eprintln!(
                        "Ignored an animation channel that did not target a joint of the same skin"
                    );
                    return None;
                }
            };
            let sampler = gltf_chan.sampler();
            let chan_reader = gltf_chan.reader(read_buf);
            let inputs = chan_reader.read_inputs()?.collect();
            let mut outputs: Vec<f32> = Vec::new();
            match chan_reader.read_outputs()? {
                Out::Translations(t) => {
                    outputs.extend(t.flat_map(|t| t.into_iter()));
                }
                Out::Rotations(r) => {
                    outputs.extend(r.into_f32().flat_map(|r| r.into_iter()));
                }
                Out::Scales(s) => {
                    outputs.extend(s.flat_map(|s| s.into_iter()));
                }
                Out::MorphTargetWeights(_) => {
                    eprintln!("Morph target animations not supported");
                    return None;
                }
            }

            Some(g_anim::Channel {
                target: match target.property() {
                    Prop::Translation => g_anim::Target::Joint {
                        id: target_joint,
                        property: g_anim::AnimatedProperty::Translation,
                    },
                    Prop::Rotation => g_anim::Target::Joint {
                        id: target_joint,
                        property: g_anim::AnimatedProperty::Rotation,
                    },
                    Prop::Scale => g_anim::Target::Joint {
                        id: target_joint,
                        property: g_anim::AnimatedProperty::Scale,
                    },
                    Prop::MorphTargetWeights => {
                        eprintln!("Morph target animations not supported");
                        return None;
                    }
                },
                ty: match target.property() {
                    Prop::Translation | Prop::Scale => g_anim::ChannelType::Vector3,
                    Prop::Rotation => g_anim::ChannelType::Rotor3,
                    Prop::MorphTargetWeights => {
                        eprintln!("Morph target animations not supported");
                        return None;
                    }
                },
                interpolation: match sampler.interpolation() {
                    Interp::Linear => g_anim::InterpolationMode::Linear,
                    Interp::Step => g_anim::InterpolationMode::Step,
                    Interp::CubicSpline => g_anim::InterpolationMode::CubicSpline,
                },
                keyframe_ts: inputs,
                data: outputs,
            })
        })
        .collect();

    g_anim::GltfAnimation::new(channels)
}
