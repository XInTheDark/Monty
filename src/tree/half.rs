use std::sync::atomic::AtomicUsize;

use crate::GameState;
use super::{Node, NodePtr};


pub struct TreeHalf {
    nodes: Vec<Node>,
    used: AtomicUsize,
    half: bool,
}

impl std::ops::Index<NodePtr> for TreeHalf {
    type Output = Node;

    fn index(&self, index: NodePtr) -> &Self::Output {
        &self.nodes[index.idx()]
    }
}

impl TreeHalf {
    pub fn new(size: usize, half: bool) -> Self {
        let mut res = Self {
            nodes: Vec::with_capacity(size),
            used: AtomicUsize::new(0),
            half,
        };

        for _ in 0..size {
            res.nodes.push(Node::new(GameState::Ongoing));
        }

        res
    }

    pub fn push_new(&self, state: GameState) -> NodePtr {
        let idx = self.used.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if idx == self.nodes.len() {
            return NodePtr::NULL;
        }

        self.nodes[idx].set_new(state);

        NodePtr::new(self.half, idx as u32)
    }
}


