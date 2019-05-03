use crate as moleengine;
use crate::{
    ecs::{event::EventQueue, system::*, IdType, Space},
    physics2d::{collision::intersection_check, Collider, Collision, RigidBody},
    util::Transform,
};

pub struct RigidBodySolver;

#[derive(ComponentFilter)]
pub struct ColliderFilter<'a> {
    #[id]
    id: IdType,
    tr: &'a Transform,
    rb: &'a mut RigidBody,
    coll: &'a Collider,
}

impl<'a> StatefulSystem<'a> for RigidBodySolver {
    type Filter = ColliderFilter<'a>;
    type State = Vec<Collision>;

    fn run_system(
        self,
        state: &mut Self::State,
        items: &mut [Self::Filter],
        _space: &Space,
        queue: &mut EventQueue,
    ) {
        let mut collisions = Vec::new();
        // ugly brute force for now
        let mut iter = items.iter();
        while let Some(o1) = iter.next() {
            for o2 in iter.clone() {
                if let Some(colls) =
                    intersection_check(o1.id, o1.tr, o1.coll, o2.id, o2.tr, o2.coll)
                {
                    // testing
                    collisions.push(colls[0]);
                    collisions.push(colls[1]);

                    queue.push(Box::new(colls[0]));
                    queue.push(Box::new(colls[1]));
                }
            }
        }

        std::mem::replace(state, collisions);
    }
}
