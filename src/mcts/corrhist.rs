use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Default, Clone)]
pub struct CorrHistEntry {
    pub delta_sum: f32,
    pub weight_sum: f32,
}

impl CorrHistEntry {
    pub fn delta(&self) -> f32 {
        if self.weight_sum.abs() < f32::EPSILON {
            0.0
        } else {
            (self.delta_sum / self.weight_sum) as f32
        }
    }
}

#[derive(Default)]
pub struct CorrHistTable {
    table: RwLock<HashMap<u64, CorrHistEntry>>,
}

impl CorrHistTable {
    pub fn new() -> Self {
        Self {
            table: RwLock::new(HashMap::new()),
        }
    }

    pub fn get_or_create(&self, ch_hash: u64) -> CorrHistEntry {
        let mut guard = self.table.write().unwrap();
        guard.entry(ch_hash).or_default().clone()
    }

    pub fn update(&self, ch_hash: u64, delta: f32, weight: f32) {
        let mut guard = self.table.write().unwrap();
        let entry = guard.entry(ch_hash).or_default();
        entry.delta_sum += delta;
        entry.weight_sum += weight;
    }
}
