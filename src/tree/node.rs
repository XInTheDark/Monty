use std::{
    ops::Add,
    sync::{
        atomic::{AtomicI32, AtomicU16, AtomicU32, AtomicU8, Ordering},
        RwLock, RwLockReadGuard, RwLockWriteGuard,
    },
};

use crate::chess::{GameState, Move};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NodePtr(u32);

impl NodePtr {
    pub const NULL: Self = Self(u32::MAX);

    pub fn is_null(self) -> bool {
        self == Self::NULL
    }

    pub fn new(half: bool, idx: u32) -> Self {
        Self((u32::from(half) << 31) | idx)
    }

    pub fn half(self) -> bool {
        self.0 & (1 << 31) > 0
    }

    pub fn idx(self) -> usize {
        (self.0 & 0x7FFFFFFF) as usize
    }

    pub fn inner(self) -> u32 {
        self.0
    }

    pub fn from_raw(inner: u32) -> Self {
        Self(inner)
    }
}

impl Add<usize> for NodePtr {
    type Output = NodePtr;

    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0 + rhs as u32)
    }
}

#[derive(Debug)]
pub struct Node {
    actions: RwLock<NodePtr>,
    num_actions: AtomicU8,
    _pad1: [u8; 63],
    state: AtomicU16,
    _pad2: [u8; 62],
    threads: AtomicU16,
    _pad3: [u8; 62],
    mov: AtomicU16,
    _pad4: [u8; 62],
    policy: AtomicU16,
    _pad5: [u8; 62],
    visits: AtomicI32,
    _pad6: [u8; 60],
    q: AtomicU32,
    _pad7: [u8; 60],
    sq_q: AtomicU32,
    _pad8: [u8; 60],
    gini_impurity: AtomicU32,
    _pad9: [u8; 60],
}

impl Node {
    pub fn new(state: GameState) -> Self {
        Node {
            actions: RwLock::new(NodePtr::NULL),
            num_actions: AtomicU8::new(0),
            _pad1: [0; 63],
            state: AtomicU16::new(u16::from(state)),
            _pad2: [0; 62],
            threads: AtomicU16::new(0),
            _pad3: [0; 62],
            mov: AtomicU16::new(0),
            _pad4: [0; 62],
            policy: AtomicU16::new(0),
            _pad5: [0; 62],
            visits: AtomicI32::new(0),
            _pad6: [0; 60],
            q: AtomicU32::new(0),
            _pad7: [0; 60],
            sq_q: AtomicU32::new(0),
            _pad8: [0; 60],
            gini_impurity: AtomicU32::new(0),
            _pad9: [0; 60],
        }
    }

    pub fn set_new(&self, mov: Move, policy: f32) {
        self.clear();
        self.mov.store(u16::from(mov), Ordering::Relaxed);
        self.set_policy(policy);
    }

    pub fn is_terminal(&self) -> bool {
        self.state() != GameState::Ongoing
    }

    pub fn num_actions(&self) -> usize {
        usize::from(self.num_actions.load(Ordering::Relaxed))
    }

    pub fn set_num_actions(&self, num: usize) {
        self.num_actions.store(num as u8, Ordering::Relaxed);
    }

    pub fn threads(&self) -> u16 {
        self.threads.load(Ordering::Relaxed)
    }

    pub fn visits(&self) -> i32 {
        self.visits.load(Ordering::Relaxed)
    }

    fn q64(&self) -> f64 {
        f64::from(self.q.load(Ordering::Relaxed)) / f64::from(u32::MAX)
    }

    pub fn q(&self) -> f32 {
        self.q64() as f32
    }

    pub fn sq_q(&self) -> f64 {
        f64::from(self.sq_q.load(Ordering::Relaxed)) / f64::from(u32::MAX)
    }

    pub fn var(&self) -> f32 {
        (self.sq_q() - self.q64().powi(2)).max(0.0) as f32
    }

    pub fn inc_threads(&self) {
        self.threads.fetch_add(1, Ordering::Relaxed);
    }

    pub fn dec_threads(&self) {
        self.threads.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn actions(&self) -> RwLockReadGuard<NodePtr> {
        self.actions.read().unwrap()
    }

    pub fn actions_mut(&self) -> RwLockWriteGuard<NodePtr> {
        self.actions.write().unwrap()
    }

    pub fn state(&self) -> GameState {
        GameState::from(self.state.load(Ordering::Relaxed))
    }

    pub fn set_state(&self, state: GameState) {
        self.state.store(u16::from(state), Ordering::Relaxed);
    }

    pub fn policy(&self) -> f32 {
        f32::from(self.policy.load(Ordering::Relaxed)) / f32::from(u16::MAX)
    }

    pub fn set_policy(&self, policy: f32) {
        self.policy
            .store((policy * f32::from(u16::MAX)) as u16, Ordering::Relaxed);
    }

    pub fn has_children(&self) -> bool {
        self.num_actions() != 0
    }

    pub fn is_not_expanded(&self) -> bool {
        self.state() == GameState::Ongoing && self.num_actions() == 0
    }

    pub fn gini_impurity(&self) -> f32 {
        f32::from_bits(self.gini_impurity.load(Ordering::Relaxed))
    }

    pub fn set_gini_impurity(&self, gini_impurity: f32) {
        self.gini_impurity
            .store(f32::to_bits(gini_impurity), Ordering::Relaxed);
    }

    pub fn clear_actions(&self) {
        *self.actions.write().unwrap() = NodePtr::NULL;
        self.num_actions.store(0, Ordering::Relaxed);
    }

    pub fn parent_move(&self) -> Move {
        Move::from(self.mov.load(Ordering::Relaxed))
    }

    pub fn copy_from(&self, other: &Self) {
        use std::sync::atomic::Ordering::Relaxed;

        self.threads.store(other.threads.load(Relaxed), Relaxed);
        self.mov.store(other.mov.load(Relaxed), Relaxed);
        self.policy.store(other.policy.load(Relaxed), Relaxed);
        self.state.store(other.state.load(Relaxed), Relaxed);
        self.gini_impurity
            .store(other.gini_impurity.load(Relaxed), Relaxed);
        self.visits.store(other.visits.load(Relaxed), Relaxed);
        self.q.store(other.q.load(Relaxed), Relaxed);
        self.sq_q.store(other.sq_q.load(Relaxed), Relaxed);
    }

    pub fn clear(&self) {
        self.clear_actions();
        self.set_state(GameState::Ongoing);
        self.set_gini_impurity(0.0);
        self.visits.store(0, Ordering::Relaxed);
        self.q.store(0, Ordering::Relaxed);
        self.sq_q.store(0, Ordering::Relaxed);
        self.threads.store(0, Ordering::Relaxed);
    }

    pub fn update(&self, result: f32) -> f32 {
        let r = f64::from(result);
        let v = f64::from(self.visits.fetch_add(1, Ordering::Relaxed));

        let q = (self.q64() * v + r) / (v + 1.0);
        let sq_q = (self.sq_q() * v + r.powi(2)) / (v + 1.0);

        self.q
            .store((q * f64::from(u32::MAX)) as u32, Ordering::Relaxed);
        self.sq_q
            .store((sq_q * f64::from(u32::MAX)) as u32, Ordering::Relaxed);

        q as f32
    }
}