use crate::{
    graphics::{util::GpuMat4, GraphicsManager},
    math::uv,
};

use itertools::izip;
use std::{mem::size_of, rc::Rc};
use zerocopy::AsBytes;

/// A hierarchy of joints used for deforming and animating meshes.
#[derive(Debug)]
pub struct Skin {
    pub(crate) root_transform: uv::Mat4,
    pub(crate) joint_set: JointSet,
    /// inverse bind matrices shared between clones of the skin
    pub(crate) inv_bind_matrices: Rc<[uv::Mat4]>,
    /// buffers and bind group to write joint matrices and skinned vertices to on the gpu,
    /// lazily initialized to avoid dependency on the mesh at creation time
    pub(crate) compute_res: Option<ComputeResources>,
}

#[derive(Debug)]
pub(crate) struct ComputeResources {
    pub vertex_buf: wgpu::Buffer,
    pub joint_matrix_buf: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl Clone for Skin {
    fn clone(&self) -> Self {
        Self {
            root_transform: self.root_transform,
            joint_set: self.joint_set.clone(),
            inv_bind_matrices: self.inv_bind_matrices.clone(),
            compute_res: None,
        }
    }
}

impl Skin {
    pub fn update_global_poses(&mut self) {
        self.joint_set.update_global_poses();
    }

    /// Update the joint matrices in this skin
    /// based on values of the joints' local poses.
    pub fn evaluate_joint_matrices(&self) -> Vec<uv::Mat4> {
        self.joint_set
            .evaluate_joint_matrices(&self.root_transform, &self.inv_bind_matrices)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct JointSet {
    pub joints: Vec<Joint>,
}

#[derive(Debug, Clone)]
pub struct Joint {
    pub name: Option<String>,
    /// index of the joint's parent joint in the skin's `joints` array
    pub parent_idx: Option<usize>,
    /// pose relative to the parent joint, updated by animations
    pub local_pose: TransformDecomp,
    /// pose in model space, as a matrix because nonuniform scalings are allowed
    /// and these can't be composed nicely outside of matrix form.
    pub global_pose: uv::Mat4,
}

impl JointSet {
    pub(crate) fn update_global_poses(&mut self) {
        // joints are given in top-down order, so we don't need recursion here,
        // the order automatically makes it so parents are evaluated first
        for joint_idx in 0..self.joints.len() {
            let local = self.joints[joint_idx].local_pose.as_matrix();
            if let Some(parent_idx) = self.joints[joint_idx].parent_idx {
                // assert to make sure the aforementioned order holds,
                // this should always be correct at least with Blender
                assert!(
                    parent_idx < joint_idx,
                    "Joints must be given in top-down order"
                );
                self.joints[joint_idx].global_pose = self.joints[parent_idx].global_pose * local;
            } else {
                self.joints[joint_idx].global_pose = local;
            }
        }
    }

    /// Compute joint matrices from the global poses,
    /// which should have been updated beforehand.
    ///
    /// `root_transform` and `inv_bind_matrices`
    /// should be those of the skin owning this joint set.
    pub(crate) fn evaluate_joint_matrices(
        &self,
        root_transform: &uv::Mat4,
        inv_bind_matrices: &[uv::Mat4],
    ) -> Vec<uv::Mat4> {
        izip!(self.joints.iter().map(|j| j.global_pose), inv_bind_matrices)
            .map(|(global_pose, inv_bind)| *root_transform * global_pose * *inv_bind)
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TransformDecomp {
    pub pos: uv::Vec3,
    pub rot: uv::Rotor3,
    pub scale: uv::Vec3,
}

impl TransformDecomp {
    pub fn from_parts((pos, rot_quat, scale): ([f32; 3], [f32; 4], [f32; 3])) -> Self {
        Self {
            pos: pos.into(),
            rot: uv::Rotor3::from_quaternion_array(rot_quat),
            scale: scale.into(),
        }
    }

    pub fn as_matrix(&self) -> uv::Mat4 {
        uv::Mat4::from_translation(self.pos)
            * self.rot.into_matrix().into_homogeneous()
            * uv::Mat4::from_nonuniform_scale(self.scale)
    }
}

//
// GPU pipeline
//

pub(crate) struct SkinPipeline {
    pipeline: wgpu::ComputePipeline,
    // layout for the bind group that binds the mesh's vertex buffer
    // and the target vertex buffer
    target_bind_group_layout: wgpu::BindGroupLayout,
}

impl SkinPipeline {
    pub fn new() -> Self {
        let device = crate::Renderer::device();

        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/skin.wgsl"));

        let target_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("skin sources and target"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        // joint matrices
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(size_of::<GpuMat4>() as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        // vertices
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(size_of::<super::Vertex>() as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        // vertex joint info
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(size_of::<super::VertexJoints>() as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        // output
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(size_of::<super::Vertex>() as _),
                        },
                        count: None,
                    },
                ],
            });

        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skin"),
            bind_group_layouts: &[&target_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("skin"),
            module: &shader,
            entry_point: "main",
            layout: Some(&pl_layout),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Self {
            pipeline,
            target_bind_group_layout,
        }
    }

    pub fn compute_skins<'pass>(
        &'pass mut self,
        pass: &mut wgpu::ComputePass<'pass>,
        manager: &'pass mut GraphicsManager,
    ) {
        let device = crate::Renderer::device();
        let queue = crate::Renderer::queue();

        // initialize gpu resources for skins that haven't had that done yet

        for (skin_id, skin) in manager
            .skins
            .iter_mut()
            .filter(|(_, skin)| skin.compute_res.is_none())
        {
            let Some(mesh) = manager
                .skin_mesh_map
                .get(skin_id)
                .and_then(|mesh_id| manager.meshes.get(*mesh_id))
            else {
                eprintln!("A mesh was removed without removing its associated skins");
                continue;
            };
            let Some(ref joints_buf) = mesh.gpu_data.joints_buf else {
                eprintln!("No joints data on a skinned mesh");
                continue;
            };

            // create the target buffer and bind group if it doesn't exist
            if skin.compute_res.is_none() {
                let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: mesh.gpu_data.vertex_buf.size(),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                let joint_matrix_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (skin.joint_set.joints.len() * size_of::<GpuMat4>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.target_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: joint_matrix_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: mesh.gpu_data.vertex_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: joints_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: vertex_buf.as_entire_binding(),
                        },
                    ],
                });

                skin.compute_res = Some(ComputeResources {
                    vertex_buf,
                    joint_matrix_buf,
                    bind_group,
                });
            }
        }

        // upload joint matrices

        for (_, skin) in manager.skins.iter() {
            let joint_matrices: Vec<GpuMat4> = skin
                .evaluate_joint_matrices()
                .into_iter()
                .map(GpuMat4::from)
                .collect();
            let res = skin.compute_res.as_ref().unwrap();
            queue.write_buffer(&res.joint_matrix_buf, 0, joint_matrices.as_bytes());
        }

        // dispatch compute work

        pass.set_pipeline(&self.pipeline);

        for (skin_id, skin) in manager.skins.iter() {
            let Some(mesh) = manager
                .skin_mesh_map
                .get(skin_id)
                .and_then(|mesh_id| manager.meshes.get(*mesh_id))
            else {
                eprintln!("A mesh was removed without removing its associated skins");
                continue;
            };

            let target = skin.compute_res.as_ref().unwrap();

            pass.set_bind_group(0, &target.bind_group, &[]);

            const WORKGROUP_SIZE: u32 = 64;
            let wg_count = (mesh.gpu_data.vertex_count / WORKGROUP_SIZE) + 1;
            pass.dispatch_workgroups(wg_count, 1, 1);
        }
    }
}
