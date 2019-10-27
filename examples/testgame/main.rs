#[macro_use]
extern crate microprofile;

mod controls;
mod recipes;
mod states;
mod test_events;

//

use glium::glutin;
use moleengine::{
    ecs, physics2d as phys,
    util::{InputCache, Transform},
    visuals_glium as vis,
};

//

pub struct DebugVisuals {
    pub contact_cache: phys::collision::ContactOutput,
    pub contact_indicator: vis::debug::ContactIndicator,
}

pub struct Resources {
    pub events: glutin::EventsLoop,
    pub space: ecs::Space,
    pub camera: vis::camera::MouseDragCamera2D,
    pub input_cache: InputCache,
    pub impulse_cache: phys::collision::ImpulseCache,
    pub debug_vis: DebugVisuals,
}

fn main() {
    microprofile::init!();
    microprofile::set_enable_all_groups!(true);

    let res = init_resources();
    states::begin(res);

    //microprofile::dump_file_immediately!("profile.html", "");
    microprofile::shutdown!();
}

pub fn init_resources() -> Resources {
    let events = unsafe { vis::Context::init() };

    let mut input_cache = InputCache::new();
    {
        use glutin::VirtualKeyCode::*;
        input_cache.track_keys(&[
            Left, Right, Down, Up, PageDown, PageUp, Escape, Return, Space, S, T, LShift,
        ]);
    }

    let space = ecs::Space::with_capacity(1000);

    let camera = vis::camera::MouseDragCamera2D::new(
        Transform::identity(),
        vis::camera::ScalingStrategy::ConstantDisplayArea {
            width: 8.0,
            height: 6.0,
        },
    );

    let impulse_cache = phys::collision::ImpulseCache::new();

    //

    let contact_cache = phys::collision::ContactOutput::new();
    let contact_indicator = vis::debug::ContactIndicator::new(&vis::Context::get().display, 50);

    Resources {
        events,
        space,
        camera,
        input_cache,
        impulse_cache,
        debug_vis: DebugVisuals {
            contact_cache,
            contact_indicator,
        },
    }
}
