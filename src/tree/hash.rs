use std::{
    array,
    sync::atomic::{AtomicU16, AtomicU32, AtomicU64, Ordering},
};

use crate::chess::Move;

use super::NodePtr;

const BUCKET_SIZE: usize = 4;

#[derive(Clone, Copy, Debug, Default)]
pub struct HashEntry {
    hash: u64,
    q: u32,
    d: u32,
    visits: u32,
    meta: u32,
    node_ptr: u64,
    structure_epoch: u32,
}

impl HashEntry {
    pub fn q(&self) -> f32 {
        self.q as f32 / u32::MAX as f32
    }

    pub fn d(&self) -> f32 {
        self.d as f32 / u32::MAX as f32
    }

    pub fn visits(&self) -> u64 {
        self.visits.into()
    }

    pub fn generation(&self) -> u16 {
        (self.meta >> 16) as u16
    }

    pub fn age(&self, current_generation: u16) -> u16 {
        current_generation.wrapping_sub(self.generation())
    }

    pub fn best_move(&self) -> Move {
        Move::from(self.meta as u16)
    }

    pub fn node_ptr(&self) -> NodePtr {
        NodePtr::from_raw(self.node_ptr)
    }

    pub fn subtree_ptr(&self, structure_epoch: u32) -> Option<NodePtr> {
        let ptr = self.node_ptr();
        if self.structure_epoch == structure_epoch && !ptr.is_null() {
            Some(ptr)
        } else {
            None
        }
    }

    fn is_empty(&self) -> bool {
        self.visits == 0
    }

    fn matches(&self, hash: u64) -> bool {
        !self.is_empty() && self.hash == hash
    }
}

struct HashEntryInternal {
    hash: AtomicU64,
    q: AtomicU32,
    d: AtomicU32,
    visits: AtomicU32,
    meta: AtomicU32,
    node_ptr: AtomicU64,
    structure_epoch: AtomicU32,
}

impl Default for HashEntryInternal {
    fn default() -> Self {
        Self {
            hash: AtomicU64::new(0),
            q: AtomicU32::new(0),
            d: AtomicU32::new(0),
            visits: AtomicU32::new(0),
            meta: AtomicU32::new(0),
            node_ptr: AtomicU64::new(NodePtr::NULL.inner()),
            structure_epoch: AtomicU32::new(0),
        }
    }
}

impl Clone for HashEntryInternal {
    fn clone(&self) -> Self {
        Self {
            hash: AtomicU64::new(self.hash.load(Ordering::Relaxed)),
            q: AtomicU32::new(self.q.load(Ordering::Relaxed)),
            d: AtomicU32::new(self.d.load(Ordering::Relaxed)),
            visits: AtomicU32::new(self.visits.load(Ordering::Relaxed)),
            meta: AtomicU32::new(self.meta.load(Ordering::Relaxed)),
            node_ptr: AtomicU64::new(self.node_ptr.load(Ordering::Relaxed)),
            structure_epoch: AtomicU32::new(self.structure_epoch.load(Ordering::Relaxed)),
        }
    }
}

impl HashEntryInternal {
    fn load(&self) -> HashEntry {
        HashEntry {
            hash: self.hash.load(Ordering::Relaxed),
            q: self.q.load(Ordering::Relaxed),
            d: self.d.load(Ordering::Relaxed),
            visits: self.visits.load(Ordering::Relaxed),
            meta: self.meta.load(Ordering::Relaxed),
            node_ptr: self.node_ptr.load(Ordering::Relaxed),
            structure_epoch: self.structure_epoch.load(Ordering::Relaxed),
        }
    }

    fn store(&self, entry: HashEntry) {
        self.q.store(entry.q, Ordering::Relaxed);
        self.d.store(entry.d, Ordering::Relaxed);
        self.visits.store(entry.visits, Ordering::Relaxed);
        self.meta.store(entry.meta, Ordering::Relaxed);
        self.node_ptr.store(entry.node_ptr, Ordering::Relaxed);
        self.structure_epoch
            .store(entry.structure_epoch, Ordering::Relaxed);
        self.hash.store(entry.hash, Ordering::Relaxed);
    }
}

struct HashBucket {
    entries: [HashEntryInternal; BUCKET_SIZE],
}

impl Default for HashBucket {
    fn default() -> Self {
        Self {
            entries: array::from_fn(|_| HashEntryInternal::default()),
        }
    }
}

impl Clone for HashBucket {
    fn clone(&self) -> Self {
        Self {
            entries: array::from_fn(|i| self.entries[i].clone()),
        }
    }
}

pub struct HashTable {
    table: Vec<HashBucket>,
    generation: AtomicU16,
    structure_epoch: AtomicU32,
}

impl HashTable {
    pub fn new(size: usize, _threads: usize) -> Self {
        let mut table = HashTable {
            table: Vec::new(),
            generation: AtomicU16::new(0),
            structure_epoch: AtomicU32::new(1),
        };

        let buckets = size.max(1).div_ceil(BUCKET_SIZE);
        table.table.resize_with(buckets, HashBucket::default);
        table
    }

    pub fn clear(&mut self, threads: usize) {
        let chunk_size = self.table.len().div_ceil(threads);

        std::thread::scope(|s| {
            for chunk in self.table.chunks_mut(chunk_size) {
                s.spawn(|| {
                    for entry in chunk.iter_mut() {
                        *entry = HashBucket::default();
                    }
                });
            }
        });

        self.generation.store(0, Ordering::Relaxed);
        self.structure_epoch.store(1, Ordering::Relaxed);
    }

    pub fn generation(&self) -> u16 {
        self.generation.load(Ordering::Relaxed)
    }

    pub fn advance_generation(&self) -> u16 {
        self.generation
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_add(1)
    }

    pub fn structure_epoch(&self) -> u32 {
        self.structure_epoch.load(Ordering::Relaxed)
    }

    pub fn advance_structure_epoch(&self) -> u32 {
        self.structure_epoch.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn bucket(&self, hash: u64) -> &HashBucket {
        let idx = hash % (self.table.len() as u64);
        &self.table[idx as usize]
    }

    fn encode_meta(generation: u16, best_move: Move) -> u32 {
        (u32::from(generation) << 16) | u32::from(u16::from(best_move))
    }

    fn replacement_score(entry: HashEntry, current_generation: u16) -> u64 {
        let age = u64::from(entry.age(current_generation));
        (age << 32) | u64::from(u32::MAX - entry.visits)
    }

    pub fn get(&self, hash: u64) -> Option<HashEntry> {
        let mut best: Option<HashEntry> = None;

        for slot in &self.bucket(hash).entries {
            let entry = slot.load();

            if !entry.matches(hash) {
                continue;
            }

            best = match best {
                Some(current)
                    if current.generation() > entry.generation()
                        || (current.generation() == entry.generation()
                            && current.visits > entry.visits) =>
                {
                    Some(current)
                }
                _ => Some(entry),
            };
        }

        best
    }

    pub fn push(
        &self,
        hash: u64,
        q: f32,
        draw: f32,
        visits: u64,
        best_move: Move,
        node_ptr: NodePtr,
    ) {
        let q_u32 = (q.clamp(0.0, 1.0) * u32::MAX as f32) as u32;
        let d_u32 = (draw.clamp(0.0, 1.0) * u32::MAX as f32) as u32;
        let visits_u32 = visits.clamp(1, u32::MAX as u64) as u32;
        let generation = self.generation();
        let structure_epoch = self.structure_epoch();
        let bucket = self.bucket(hash);

        let mut target_idx = 0usize;
        let mut exact_match = None;
        let mut best_replace_score = u64::MIN;

        for (idx, slot) in bucket.entries.iter().enumerate() {
            let entry = slot.load();

            if entry.matches(hash) {
                exact_match = Some((idx, entry));
                break;
            }

            if entry.is_empty() {
                target_idx = idx;
                break;
            }

            let score = Self::replacement_score(entry, generation);
            if score >= best_replace_score {
                best_replace_score = score;
                target_idx = idx;
            }
        }

        let new_entry = if let Some((idx, existing)) = exact_match {
            target_idx = idx;

            let keep_existing_stats =
                existing.generation() == generation && existing.visits > visits_u32;
            let chosen_best_move = if best_move == Move::NULL {
                existing.best_move()
            } else {
                best_move
            };
            let chosen_node_ptr = if node_ptr.is_null() {
                existing.node_ptr()
            } else {
                node_ptr
            };
            let chosen_structure_epoch = if node_ptr.is_null() {
                existing.structure_epoch
            } else {
                structure_epoch
            };

            HashEntry {
                hash,
                q: if keep_existing_stats {
                    existing.q
                } else {
                    q_u32
                },
                d: if keep_existing_stats {
                    existing.d
                } else {
                    d_u32
                },
                visits: existing.visits.max(visits_u32),
                meta: Self::encode_meta(generation, chosen_best_move),
                node_ptr: chosen_node_ptr.inner(),
                structure_epoch: chosen_structure_epoch,
            }
        } else {
            HashEntry {
                hash,
                q: q_u32,
                d: d_u32,
                visits: visits_u32,
                meta: Self::encode_meta(generation, best_move),
                node_ptr: node_ptr.inner(),
                structure_epoch,
            }
        };

        bucket.entries[target_idx].store(new_entry);
    }
}

#[cfg(test)]
mod tests {
    use super::HashTable;
    use crate::chess::Move;
    use crate::tree::NodePtr;

    #[test]
    fn exact_entries_keep_stronger_stats_but_merge_metadata() {
        let table = HashTable::new(4, 1);
        let hash = 0x1234_5678_90ab_cdef;
        let old_ptr = NodePtr::new(false, 7);
        let new_ptr = NodePtr::new(true, 11);

        table.push(hash, 0.75, 0.10, 32, Move::from(12), old_ptr);
        table.push(hash, 0.25, 0.20, 8, Move::NULL, new_ptr);

        let entry = table.get(hash).unwrap();
        assert_eq!(entry.visits(), 32);
        assert_eq!(entry.best_move(), Move::from(12));
        assert_eq!(entry.node_ptr(), new_ptr);
        assert!((entry.q() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn subtree_pointer_respects_structure_epoch() {
        let table = HashTable::new(4, 1);
        let hash = 0xfeed_face_dead_beef;
        let ptr = NodePtr::new(false, 3);

        table.push(hash, 0.5, 0.1, 1, Move::NULL, ptr);
        let entry = table.get(hash).unwrap();
        let epoch = table.structure_epoch();

        assert_eq!(entry.subtree_ptr(epoch), Some(ptr));
        table.advance_structure_epoch();
        assert_eq!(entry.subtree_ptr(table.structure_epoch()), None);
    }

    #[test]
    fn bucketed_lookup_survives_collisions() {
        let table = HashTable::new(4, 1);
        let hashes = [0, 4, 8, 12];

        for (idx, hash) in hashes.into_iter().enumerate() {
            table.push(
                hash,
                0.1 * idx as f32,
                0.05,
                (idx + 1) as u64,
                Move::from(idx as u16),
                NodePtr::new(false, idx),
            );
        }

        for (idx, hash) in hashes.into_iter().enumerate() {
            let entry = table.get(hash).unwrap();
            assert_eq!(entry.visits(), (idx + 1) as u64);
            assert_eq!(entry.best_move(), Move::from(idx as u16));
        }
    }
}
