use super::event::EventQueue;
use super::space::Space;

pub(crate) use crate as moleengine; // the macros use the form moleengine::* for moleengine types
pub use hibitset;
pub use moleengine_ecs_codegen::*;

/// A System can perform arbitrary operations on game objects with desired associated Components,
/// which are defined as a ComponentFilter.
/// Systems can be given context information in the form of `self`, but in contrast to StatefulSystems,
/// this information is not allowed to persist between updates. For this reason `self` is consumed by `run_system`.
/// They are executed within a Space with Space::(try_)run_system::<Type>().
pub trait System<'a> {
    type Filter: ComponentFilter<'a>;
    fn run_system(self, items: &mut [Self::Filter], space: &Space, queue: &mut EventQueue);
}

/// A simpler System interface useful to reduce boilerplate
/// when implementing Systems which only use one filter and don't produce events.
pub trait SimpleSystem<'a> {
    type Filter: ComponentFilter<'a>;
    fn run_system(self, items: &mut [Self::Filter]);
}

impl<'a, S: SimpleSystem<'a>> System<'a> for S {
    type Filter = S::Filter;
    fn run_system(self, items: &mut [Self::Filter], _s: &Space, _q: &mut EventQueue) {
        <Self as SimpleSystem>::run_system(self, items);
    }
}

/// A set of Components that knows how to extract itself from a Space.
/// These do not need to be implemented by hand - there is a derive macro available
/// in the moleengine_ecs_codegen crate.
/// # Example
/// ```
/// #[derive(ComponentFilter)]
/// pub struct PosVel<'a> {
///     #[id] id: IdType,
///     position: &'a mut Position,
///     velocity: &'a Velocity,
/// }
/// ```
pub trait ComponentFilter<'a>: Sized {
    fn run_filter(space: &Space, f: impl FnOnce(&mut [Self])) -> Option<()>;
}
