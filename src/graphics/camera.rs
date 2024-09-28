use crate::{
    graphics::util::{GpuMat4, GpuVec2Padded},
    math::{self as m, uv},
};
use std::{mem::size_of, sync::OnceLock};
use zerocopy::{AsBytes, FromBytes};

mod mouse_drag;
pub use mouse_drag::MouseDragCameraController;

/// Bind group layout for camera uniforms
/// created when the first camera is made
static BIND_GROUP_LAYOUT: OnceLock<wgpu::BindGroupLayout> = OnceLock::new();

/// A camera determines the area of space to draw when rendering.
#[derive(Debug)]
pub struct Camera {
    /// Width of the minimum seen area at zoom level 0 in world units. Default: 20
    pub view_width: f32,
    /// Height of the minimum seen area at zoom level 0 in world units. Default: 10
    pub view_height: f32,
    /// Pose of the camera in 3D world space.
    pub pose: uv::Isometry3,
    /// Level of magnification from the default `view_width`, `view_height`.
    pub zoom: f32,
    /// Near plane of the orthographic projection. Default: -1000
    pub z_near: f32,
    /// Far plane of the orthographic projection. Default: 1000
    pub z_far: f32,
    // internal GPU resources
    uniform_buf: wgpu::Buffer,
    pub(crate) bind_group: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, AsBytes, FromBytes)]
pub(crate) struct CameraUniforms {
    view_proj: GpuMat4,
    view: GpuMat4,
    viewport_size_world: GpuVec2Padded,
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

impl Camera {
    pub fn new() -> Self {
        let device = crate::Renderer::device();

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            size: size_of::<CameraUniforms>() as _,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            label: Some("mesh camera"),
            mapped_at_creation: false,
        });

        let bind_group_layout = Self::bind_group_layout();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera"),
            layout: bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        Self {
            view_width: 20.,
            view_height: 10.,
            pose: Default::default(),
            zoom: 1.,
            z_near: -1000.,
            z_far: 1000.,
            uniform_buf,
            bind_group,
        }
    }

    pub(crate) fn bind_group_layout<'a>() -> &'a wgpu::BindGroupLayout {
        BIND_GROUP_LAYOUT.get_or_init(|| {
            let device = crate::Renderer::device();
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(size_of::<CameraUniforms>() as _),
                    },
                    count: None,
                }],
                label: Some("camera"),
            })
        })
    }

    /// Upload the current state of the camera to the GPU.
    /// Call at the start of a frame.
    pub fn upload(&mut self) {
        let queue = crate::Renderer::queue();

        let view = self.view_matrix();
        let proj = self.projection_matrix();
        let view_proj = proj * view;
        let unif = CameraUniforms {
            view_proj: view_proj.into(),
            view: view.into(),
            viewport_size_world: self.visible_area_size().into(),
        };
        queue.write_buffer(&self.uniform_buf, 0, unif.as_bytes());
    }

    /// Viewport pixels per world unit, taking into consideration zoom level.
    pub(crate) fn pixels_per_world_unit(&self, viewport_size: (u32, u32)) -> f32 {
        let (vp_w, vp_h) = viewport_size;
        (vp_w as f32 / self.view_width).min(vp_h as f32 / self.view_height) * self.zoom
    }

    /// Compute the area seen by this camera,
    /// taking into account zoom level and window aspect ratio.
    pub fn visible_area_size(&self) -> m::Vec2 {
        let window = crate::Renderer::window();
        let win_size = window.inner_size();
        let aspect_ratio = win_size.width as f32 / win_size.height as f32;
        let target_ratio = self.view_width / self.view_height;
        if aspect_ratio <= target_ratio {
            m::Vec2::new(self.view_width, self.view_width / aspect_ratio) / self.zoom
        } else {
            m::Vec2::new(self.view_height * aspect_ratio, self.view_height) / self.zoom
        }
    }

    /// The matrix transforming coordinates from world space to camera space.
    #[inline]
    pub fn view_matrix(&self) -> uv::Mat4 {
        self.pose.inversed().into_homogeneous_matrix()
    }

    /// The orthographic projection matrix used by this camera.
    pub fn projection_matrix(&self) -> uv::Mat4 {
        let window = crate::Renderer::window();
        // This assumes that the viewport being drawn to is the game window,
        // which is currently the only target you can draw to.
        // if this changes, we'll have to rethink this architecture -
        // can one camera be used to draw to multiple viewports?
        // if so, it would need more than one uniform buffer.
        // alternatively the viewport size could be a different uniform
        // instead of being packaged into the projection matrix
        let viewport_size = window.inner_size().into();
        let ppwu = self.pixels_per_world_unit(viewport_size);

        let z_range_size = self.z_far - self.z_near;
        // orthographic projection from starframe's left-handed space
        // to the also left-handed wgpu device coordinates
        uv::Mat4::new(
            uv::Vec4::new(ppwu * 2. / viewport_size.0 as f32, 0., 0., 0.),
            uv::Vec4::new(0., ppwu * 2. / viewport_size.1 as f32, 0., 0.),
            uv::Vec4::new(0., 0., 1. / z_range_size, 0.),
            uv::Vec4::new(0., 0., -self.z_near / z_range_size, 1.),
        )
    }

    /// Transform a point from screen space into world space.
    ///
    /// This expects that the camera has not been rotated outside of the xy plane.
    /// Results will be incorrect otherwise.
    pub fn point_screen_to_world(&self, point_screen: m::Vec2) -> m::Vec2 {
        let window = crate::Renderer::window();
        let viewport_size = window.inner_size().into();

        let ppwu = self.pixels_per_world_unit(viewport_size);
        let half_vp_diag = m::Vec2::new(viewport_size.0 as f32 / 2., viewport_size.1 as f32 / 2.);
        let point_screen_wrt_center = {
            let p = point_screen - half_vp_diag;
            m::Vec2::new(p.x, -p.y) / ppwu
        };

        self.pose_as_2d() * point_screen_wrt_center
    }

    /// Transform a displacement vector from screen space into world space.
    ///
    /// This expects that the camera has not been rotated outside of the xy plane.
    /// Results will be incorrect otherwise.
    pub fn vector_screen_to_world(&self, vec_screen: m::Vec2) -> m::Vec2 {
        let window = crate::Renderer::window();
        let viewport_size = window.inner_size().into();
        let ppwu = self.pixels_per_world_unit(viewport_size);
        m::Vec2::new(vec_screen.x, -vec_screen.y) / ppwu
    }

    fn pose_as_2d(&self) -> uv::Isometry2 {
        uv::Isometry2::new(
            uv::Vec2::new(self.pose.translation.x, self.pose.translation.y),
            uv::Rotor2::new(
                self.pose.rotation.s,
                uv::Bivec2::new(self.pose.rotation.bv.xy),
            ),
        )
    }
}
