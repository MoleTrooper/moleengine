//! A Bounding Volume Hierarchy implementation
//! for speeding up collision detection and other spatial queries.

use crate::{
    math::uv,
    physics::{
        collision::{query::ray_aabb, Ray, AABB},
        ColliderKey,
    },
};

use std::collections::BinaryHeap;

//
// Internal types
//

#[derive(Clone, Copy, Debug)]
struct Node {
    aabb: AABB,
    kind: NodeKind,
}

#[derive(Clone, Copy, Debug)]
enum NodeKind {
    Branch { left: usize, right: usize },
    Leaf { coll_key: ColliderKey },
}

/// A "call stack" for efficient recursion through the tree.
#[derive(Clone, Debug)]
pub struct Stack(Vec<usize>);

/// Like a Stack, but ordered by reverse distance
/// for traversing the BVH in spatial order along a ray.
#[derive(Clone, Debug)]
pub struct RayStack(BinaryHeap<RayStackEntry>);

impl RayStack {
    fn push(&mut self, node_idx: usize, distance: f64) {
        self.0.push(RayStackEntry {
            node_idx,
            t: distance,
        });
    }

    fn pop(&mut self) -> Option<RayStackEntry> {
        self.0.pop()
    }

    fn peek_t(&self) -> Option<f64> {
        self.0.peek().map(|entry| entry.t)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RayStackEntry {
    node_idx: usize,
    t: f64,
}
impl Eq for RayStackEntry {}

impl Ord for RayStackEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .t
            .partial_cmp(&self.t)
            .expect("Bug in AABB sweep code")
    }
}
impl PartialOrd for RayStackEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

//
// BVH itself
//

/// A Bounding Volume Hierarchy implemented as an
/// incrementally constructed binary AABB tree.
#[derive(Clone, Debug)]
pub struct Bvh {
    nodes: Vec<Node>,
    /// Single stack that is kept around so that we don't need to
    /// allocate a separate one for every recursive traversal.
    shared_stack: Stack,
    shared_ray_stack: RayStack,
}

impl Bvh {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            shared_stack: Stack(Vec::new()),
            shared_ray_stack: RayStack(BinaryHeap::new()),
        }
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    pub fn insert(&mut self, coll_key: ColliderKey, aabb: AABB) {
        let new_node = Node {
            aabb,
            kind: NodeKind::Leaf { coll_key },
        };

        if self.nodes.is_empty() {
            self.nodes.push(new_node);
            return;
        }

        // traverse the tree and find a nice spot to put the new thing

        let mut curr_node_idx = 0;
        loop {
            let curr_node = self.nodes[curr_node_idx];
            match curr_node.kind {
                NodeKind::Branch { left, right } => {
                    // recurse down whichever path would create the smaller area
                    // and update the branch node to contain the union of its old
                    // aabb and the new one

                    let left_union = aabb.union(&self.nodes[left].aabb);
                    let right_union = aabb.union(&self.nodes[right].aabb);

                    if left_union.area() <= right_union.area() {
                        self.nodes[curr_node_idx].aabb = left_union.union(&curr_node.aabb);
                        if matches!(self.nodes[left].kind, NodeKind::Branch { .. }) {
                            // if the node below is a branch node, update it now
                            // so we don't have to recompute the union on the next level down
                            self.nodes[left].aabb = left_union;
                        }
                        curr_node_idx = left;
                    } else {
                        self.nodes[curr_node_idx].aabb = right_union.union(&curr_node.aabb);
                        if matches!(self.nodes[right].kind, NodeKind::Branch { .. }) {
                            self.nodes[right].aabb = right_union;
                        }
                        curr_node_idx = right;
                    }

                    if curr_node_idx == 0 {
                        // if we're at the root node, the union computations for branches (above)
                        // are never done so we need to update the root node like this
                        self.nodes[curr_node_idx].aabb = curr_node.aabb.union(&new_node.aabb);
                    }
                }
                NodeKind::Leaf { .. } => {
                    // add a branch where this leaf was and push the leaf to the end,
                    // paired with the new leaf we're currently adding

                    self.nodes.push(curr_node);
                    self.nodes.push(new_node);
                    self.nodes[curr_node_idx] = Node {
                        aabb: curr_node.aabb.union(&new_node.aabb),
                        kind: NodeKind::Branch {
                            left: self.nodes.len() - 2,
                            right: self.nodes.len() - 1,
                        },
                    };
                    return;
                }
            }
        }
    }

    pub fn test_aabb(&mut self, aabb: AABB) -> AABBIter<'_> {
        AABBIter {
            aabb,
            stack: &mut self.shared_stack,
            nodes: &self.nodes,
            // explicitly handle the special cases of zero or one nodes in the tree,
            // the iterator does not consider these
            next_node: if self.nodes.is_empty() {
                None
            } else if self.nodes.len() == 1 {
                if self.nodes[0].aabb.intersection(&aabb).is_some() {
                    Some(0)
                } else {
                    None
                }
            } else {
                Some(0)
            },
        }
    }

    pub fn test_point(&mut self, point: uv::DVec2) -> PointIter<'_> {
        PointIter {
            point,
            stack: &mut self.shared_stack,
            nodes: &self.nodes,
            next_node: if self.nodes.is_empty() {
                None
            } else if self.nodes.len() == 1 {
                if self.nodes[0].aabb.contains_point(point) {
                    Some(0)
                } else {
                    None
                }
            } else {
                Some(0)
            },
        }
    }

    pub fn sweep_aabb(&mut self, box_half_size: f64, ray: Ray, max_t: f64) -> AABBSweep<'_> {
        AABBSweep {
            ray,
            box_half_size,
            max_t,
            stack: &mut self.shared_ray_stack,
            nodes: &self.nodes,
            next_node: if self.nodes.is_empty() {
                None
            } else if self.nodes.len() == 1 {
                ray_aabb(ray, self.nodes[0].aabb.padded(box_half_size)).and_then(|t| {
                    if t <= max_t {
                        Some(RayStackEntry { node_idx: 0, t })
                    } else {
                        None
                    }
                })
            } else {
                Some(RayStackEntry {
                    node_idx: 0,
                    t: 0.0,
                })
            },
        }
    }

    // Generate a list of AABBS for debug drawing.
    /// (currently unused as the old visualizing pipelines don't work anymore
    /// and I haven't needed them since)
    #[allow(dead_code)]
    pub(crate) fn all_nodes(&self) -> Vec<NodeInfo> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        // list of right children of branch nodes we've only been to the left child of (DFS)
        let mut stack: Vec<(usize, usize)> = Vec::new();
        // collected list of nodes to return
        let mut nodes = Vec::new();
        let mut curr_node_idx = 0;
        let mut curr_depth = 0;
        'dfs: loop {
            let curr_node = self.nodes[curr_node_idx];
            nodes.push(NodeInfo {
                aabb: curr_node.aabb,
                depth: curr_depth,
            });
            match curr_node.kind {
                NodeKind::Branch { left, right } => {
                    stack.push((right, curr_depth + 1));
                    curr_node_idx = left;
                    curr_depth += 1;
                }
                NodeKind::Leaf { .. } => match stack.pop() {
                    Some((next, depth)) => {
                        curr_node_idx = next;
                        curr_depth = depth;
                    }
                    None => break 'dfs,
                },
            }
        }
        nodes
    }
}

#[allow(dead_code)]
pub(crate) struct NodeInfo {
    pub aabb: AABB,
    pub depth: usize,
}

//
// Iterators
//

// None of these handle the cases of zero or one nodes in the tree.
// Remember to do those in the BVH methods that create these.

/// An iterator that yields every collider that may intersect with a given AABB.
#[derive(Debug)]
pub struct AABBIter<'a> {
    aabb: AABB,
    stack: &'a mut Stack,
    nodes: &'a [Node],
    next_node: Option<usize>,
}

impl Iterator for AABBIter<'_> {
    type Item = ColliderKey;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next_node = self.next_node?;

            match self.nodes[next_node].kind {
                NodeKind::Branch { left, right } => {
                    match (
                        self.aabb.intersection(&self.nodes[left].aabb),
                        self.aabb.intersection(&self.nodes[right].aabb),
                    ) {
                        (Some(_), Some(_)) => {
                            // need to visit both children, push to stack to return to later
                            self.stack.0.push(right);
                            self.next_node = Some(left);
                        }
                        (Some(_), None) => {
                            self.next_node = Some(left);
                        }
                        (None, Some(_)) => {
                            self.next_node = Some(right);
                        }
                        (None, None) => {
                            // nothing below this, return back up the stack
                            self.next_node = self.stack.0.pop();
                        }
                    }
                }
                NodeKind::Leaf { coll_key } => {
                    self.next_node = self.stack.0.pop();
                    return Some(coll_key);
                }
            }
        }
    }
}

impl Drop for AABBIter<'_> {
    fn drop(&mut self) {
        // clear the stack on drop; it may not be empty
        // if the iteration didn't finish
        self.stack.0.clear();
    }
}

/// An iterator that yields every collider that may intersect with a given point.
#[derive(Debug)]
pub struct PointIter<'a> {
    point: uv::DVec2,
    stack: &'a mut Stack,
    nodes: &'a [Node],
    next_node: Option<usize>,
}

impl Iterator for PointIter<'_> {
    type Item = ColliderKey;

    fn next(&mut self) -> Option<Self::Item> {
        // exact same thing as above but with a point instead of aabb
        loop {
            let next_node = self.next_node?;

            match self.nodes[next_node].kind {
                NodeKind::Branch { left, right } => {
                    match (
                        self.nodes[left].aabb.contains_point(self.point),
                        self.nodes[right].aabb.contains_point(self.point),
                    ) {
                        (true, true) => {
                            self.stack.0.push(right);
                            self.next_node = Some(left);
                        }
                        (true, false) => {
                            self.next_node = Some(left);
                        }
                        (false, true) => {
                            self.next_node = Some(right);
                        }
                        (false, false) => {
                            self.next_node = self.stack.0.pop();
                        }
                    }
                }
                NodeKind::Leaf { coll_key } => {
                    self.next_node = self.stack.0.pop();
                    return Some(coll_key);
                }
            }
        }
    }
}

impl Drop for PointIter<'_> {
    fn drop(&mut self) {
        self.stack.0.clear();
    }
}

/// An iterator that sweeps an AABB along a ray and returns every bounding volume intersected.
#[derive(Debug)]
pub struct AABBSweep<'a> {
    // params
    ray: Ray,
    box_half_size: f64,
    max_t: f64,
    // state
    stack: &'a mut RayStack,
    nodes: &'a [Node],
    next_node: Option<RayStackEntry>,
}

#[derive(Clone, Copy, Debug)]
pub struct SweepItem {
    pub t: f64,
    pub coll_key: ColliderKey,
}

impl Iterator for AABBSweep<'_> {
    type Item = SweepItem;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next_node = self.next_node?;

            match self.nodes[next_node.node_idx].kind {
                NodeKind::Branch { left, right } => {
                    // querying ray against padded AABB is the same as
                    // sweeping AABB against the original AABB
                    // (it's the Minkowski sum thing used with spherecasts, but in Manhattan distance!)
                    let left_aabb = self.nodes[left].aabb.padded(self.box_half_size);
                    let right_aabb = self.nodes[right].aabb.padded(self.box_half_size);
                    match (
                        ray_aabb(self.ray, left_aabb),
                        ray_aabb(self.ray, right_aabb),
                    ) {
                        (Some(t_l), Some(t_r)) => {
                            let (closer_t, closer_idx, farther_t, farther_idx) = if t_l <= t_r {
                                (t_l, left, t_r, right)
                            } else {
                                (t_r, right, t_l, left)
                            };

                            if closer_t >= self.max_t {
                                // reached end of ray, go back in the stack
                                self.next_node = self.stack.pop();
                                continue;
                            }
                            if farther_t < self.max_t {
                                // both children are along the ray, stack the farther one
                                // to return to later
                                self.stack.push(farther_idx, farther_t);
                            }
                            // pick the closer one from the top of the stack or the closer child,
                            // so that we traverse the whole tree in spatial order
                            if matches!(self.stack.peek_t(), Some(nearest_t) if nearest_t < closer_t)
                            {
                                self.next_node = self.stack.pop();
                                self.stack.push(closer_idx, closer_t);
                            } else {
                                self.next_node = Some(RayStackEntry {
                                    t: closer_t,
                                    node_idx: closer_idx,
                                });
                            }
                        }
                        (Some(t_l), None) => {
                            if t_l >= self.max_t {
                                self.next_node = self.stack.pop();
                                continue;
                            }

                            if matches!(self.stack.peek_t(), Some(nearest_t) if nearest_t < t_l) {
                                self.next_node = self.stack.pop();
                                self.stack.push(left, t_l);
                            } else {
                                self.next_node = Some(RayStackEntry {
                                    t: t_l,
                                    node_idx: left,
                                });
                            }
                        }
                        (None, Some(t_r)) => {
                            if t_r >= self.max_t {
                                self.next_node = self.stack.pop();
                                continue;
                            }

                            if matches!(self.stack.peek_t(), Some(nearest_t) if nearest_t < t_r) {
                                self.next_node = self.stack.pop();
                                self.stack.push(right, t_r);
                            } else {
                                self.next_node = Some(RayStackEntry {
                                    t: t_r,
                                    node_idx: right,
                                });
                            }
                        }
                        (None, None) => {
                            self.next_node = self.stack.pop();
                        }
                    }
                }
                NodeKind::Leaf { coll_key } => {
                    self.next_node = self.stack.pop();
                    return Some(SweepItem {
                        t: next_node.t,
                        coll_key,
                    });
                }
            }
        }
    }
}

impl Drop for AABBSweep<'_> {
    fn drop(&mut self) {
        self.stack.0.clear();
    }
}
