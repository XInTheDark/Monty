use std::{
    array,
    sync::atomic::{AtomicU16, AtomicU32, AtomicU64, Ordering},
};

use crate::chess::Move;

const BUCKET_SIZE: usize = 4;

#[derive(Clone, Copy, Debug, Default)]
pub struct HashEntry {
    hash: u64,
    q: u32,
    d: u32,
    visits: u32,
    best_child_visits: u32,
    meta: u32,
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

    pub fn best_child_visits(&self) -> u64 {
        self.best_child_visits.into()
    }
}

struct HashEntryInternal {
    hash: AtomicU64,
    q: AtomicU32,
    d: AtomicU32,
    visits: AtomicU32,
    best_child_visits: AtomicU32,
    meta: AtomicU32,
}

impl Default for HashEntryInternal {
    fn default() -> Self {
        Self {
            hash: AtomicU64::new(0),
            q: AtomicU32::new(0),
            d: AtomicU32::new(0),
            visits: AtomicU32::new(0),
            best_child_visits: AtomicU32::new(0),
            meta: AtomicU32::new(0),
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
            best_child_visits: AtomicU32::new(self.best_child_visits.load(Ordering::Relaxed)),
            meta: AtomicU32::new(self.meta.load(Ordering::Relaxed)),
        }
    }
}

impl HashEntryInternal {
    fn hash(&self) -> u64 {
        self.hash.load(Ordering::Relaxed)
    }

    fn visits(&self) -> u32 {
        self.visits.load(Ordering::Relaxed)
    }

    fn meta(&self) -> u32 {
        self.meta.load(Ordering::Relaxed)
    }

    fn load(&self) -> HashEntry {
        HashEntry {
            hash: self.hash(),
            q: self.q.load(Ordering::Relaxed),
            d: self.d.load(Ordering::Relaxed),
            visits: self.visits(),
            best_child_visits: self.best_child_visits.load(Ordering::Relaxed),
            meta: self.meta(),
        }
    }

    fn store(&self, entry: HashEntry) {
        self.q.store(entry.q, Ordering::Relaxed);
        self.d.store(entry.d, Ordering::Relaxed);
        self.visits.store(entry.visits, Ordering::Relaxed);
        self.best_child_visits
            .store(entry.best_child_visits, Ordering::Relaxed);
        self.meta.store(entry.meta, Ordering::Relaxed);
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
}

impl HashTable {
    pub fn new(size: usize, _threads: usize) -> Self {
        let mut table = HashTable {
            table: Vec::new(),
            generation: AtomicU16::new(0),
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
    }

    pub fn generation(&self) -> u16 {
        self.generation.load(Ordering::Relaxed)
    }

    pub fn advance_generation(&self) -> u16 {
        self.generation
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_add(1)
    }

    fn bucket(&self, hash: u64) -> &HashBucket {
        let idx = hash % (self.table.len() as u64);
        &self.table[idx as usize]
    }

    fn encode_meta(generation: u16, best_move: Move) -> u32 {
        (u32::from(generation) << 16) | u32::from(u16::from(best_move))
    }

    fn replacement_score(generation: u16, visits: u32, current_generation: u16) -> u64 {
        let age = u64::from(current_generation.wrapping_sub(generation));
        (age << 32) | u64::from(u32::MAX - visits)
    }

    pub fn get(&self, hash: u64) -> Option<HashEntry> {
        let bucket = self.bucket(hash);
        let mut best_idx = None;
        let mut best_generation = 0u16;
        let mut best_visits = 0u32;

        for (idx, slot) in bucket.entries.iter().enumerate() {
            let visits = slot.visits();
            if visits == 0 {
                break;
            }

            if slot.hash() != hash {
                continue;
            }

            let generation = (slot.meta() >> 16) as u16;
            if best_idx.is_none()
                || best_generation < generation
                || (best_generation == generation && best_visits < visits)
            {
                best_idx = Some(idx);
                best_generation = generation;
                best_visits = visits;
            }
        }

        best_idx.map(|idx| bucket.entries[idx].load())
    }

    pub fn push(
        &self,
        hash: u64,
        q: f32,
        draw: f32,
        visits: u64,
        best_move: Move,
        best_child_visits: u64,
    ) {
        let q_u32 = (q.clamp(0.0, 1.0) * u32::MAX as f32) as u32;
        let d_u32 = (draw.clamp(0.0, 1.0) * u32::MAX as f32) as u32;
        let visits_u32 = visits.clamp(1, u32::MAX as u64) as u32;
        let best_child_visits_u32 = best_child_visits.min(u32::MAX as u64) as u32;
        let generation = self.generation();
        let bucket = self.bucket(hash);

        let mut target_idx = 0usize;
        let mut exact_match = None;
        let mut best_replace_score = u64::MIN;

        for (idx, slot) in bucket.entries.iter().enumerate() {
            let visits = slot.visits();
            if visits == 0 {
                target_idx = idx;
                break;
            }

            if slot.hash() == hash {
                exact_match = Some((idx, slot.load()));
                break;
            }

            let score = Self::replacement_score((slot.meta() >> 16) as u16, visits, generation);
            if score >= best_replace_score {
                best_replace_score = score;
                target_idx = idx;
            }
        }

        let new_entry = if let Some((idx, existing)) = exact_match {
            target_idx = idx;

            let keep_existing_stats =
                existing.generation() == generation && existing.visits > visits_u32;
            let keep_existing_best = best_move == Move::NULL
                || best_child_visits_u32 == 0
                || (existing.best_move() != Move::NULL
                    && existing.best_child_visits >= best_child_visits_u32);
            let (chosen_best_move, chosen_best_child_visits) = if keep_existing_best {
                (existing.best_move(), existing.best_child_visits)
            } else {
                (best_move, best_child_visits_u32)
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
                best_child_visits: chosen_best_child_visits,
                meta: Self::encode_meta(generation, chosen_best_move),
            }
        } else {
            HashEntry {
                hash,
                q: q_u32,
                d: d_u32,
                visits: visits_u32,
                best_child_visits: best_child_visits_u32,
                meta: Self::encode_meta(generation, best_move),
            }
        };

        bucket.entries[target_idx].store(new_entry);
    }
}
