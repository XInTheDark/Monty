use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, Default)]
pub struct HashEntry {
    hash: u16,
    q: u16,
}

impl HashEntry {
    pub fn q(&self) -> f32 {
        f32::from(self.q) / f32::from(u16::MAX)
    }
}

#[derive(Default)]
struct HashEntryInternal(AtomicU32);

impl Clone for HashEntryInternal {
    fn clone(&self) -> Self {
        Self(AtomicU32::new(self.0.load(Ordering::Relaxed)))
    }
}

impl From<&HashEntryInternal> for HashEntry {
    fn from(value: &HashEntryInternal) -> Self {
        unsafe { std::mem::transmute(value.0.load(Ordering::Relaxed)) }
    }
}

impl From<HashEntry> for u32 {
    fn from(value: HashEntry) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

pub struct HashTable {
    table: Vec<HashEntryInternal>,
}

impl HashTable {
    pub fn new(size: usize, threads: usize) -> Self {
        let chunk_size = (size + threads - 1) / threads;

        let mut table = HashTable { table: Vec::new() };
        table.table.reserve_exact(size);

        unsafe {
            use std::mem::{size_of, MaybeUninit};
            let ptr = table.table.as_mut_ptr().cast();
            let uninit: &mut [MaybeUninit<u8>] =
                std::slice::from_raw_parts_mut(ptr, size * size_of::<HashEntryInternal>());

            std::thread::scope(|s| {
                for chunk in uninit.chunks_mut(chunk_size) {
                    s.spawn(|| {
                        chunk.as_mut_ptr().write_bytes(0, chunk.len());
                    });
                }
            });

            table.table.set_len(size);
        }

        table
    }

    pub fn clear(&mut self, threads: usize) {
        let chunk_size = (self.table.len() + threads - 1) / threads;

        std::thread::scope(|s| {
            for chunk in self.table.chunks_mut(chunk_size) {
                s.spawn(|| {
                    for entry in chunk.iter_mut() {
                        *entry = HashEntryInternal::default();
                    }
                });
            }
        });
    }

    pub fn fetch(&self, hash: u64) -> HashEntry {
        let idx = hash % (self.table.len() as u64);
        HashEntry::from(&self.table[idx as usize])
    }

    fn key(hash: u64) -> u16 {
        (hash >> 48) as u16
    }

    pub fn get(&self, hash: u64) -> Option<HashEntry> {
        let entry = self.fetch(hash);

        if entry.hash == Self::key(hash) {
            Some(entry)
        } else {
            None
        }
    }

    pub fn push(&self, hash: u64, q: f32) {
        let idx = hash % (self.table.len() as u64);

        let entry = HashEntry {
            hash: Self::key(hash),
            q: (q * f32::from(u16::MAX)) as u16,
        };

        self.table[idx as usize]
            .0
            .store(u32::from(entry), Ordering::Relaxed)
    }
}

#[derive(Default, Clone, Copy)]
pub struct CorrectionHistoryEntry {
    pub value: f32,
    pub visits: u32,
}

impl CorrectionHistoryEntry {
    pub fn new(value: f32) -> Self {
        Self { value, visits: 0 }
    }

    pub fn delta(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value / (self.visits as f32)
        }
    }
}

struct CorrectionHistoryEntryInternal(AtomicU64);

impl Clone for CorrectionHistoryEntryInternal {
    fn clone(&self) -> Self {
        Self(AtomicU64::new(self.0.load(Ordering::Relaxed)))
    }
}

impl From<&CorrectionHistoryEntryInternal> for CorrectionHistoryEntry {
    fn from(value: &CorrectionHistoryEntryInternal) -> Self {
        unsafe { std::mem::transmute(value.0.load(Ordering::Relaxed)) }
    }
}

impl From<CorrectionHistoryEntry> for u64 {
    fn from(value: CorrectionHistoryEntry) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

pub struct CorrectionHistoryHashTable {
    table: Vec<CorrectionHistoryEntryInternal>,
}

const CORRECTION_HISTORY_SIZE: u64 = 16384;

impl CorrectionHistoryHashTable {
    pub fn new() -> Self {
        let table = vec![
            CorrectionHistoryEntryInternal(AtomicU64::new(0));
            CORRECTION_HISTORY_SIZE as usize
        ];
        CorrectionHistoryHashTable { table }
    }

    pub fn get(&self, key: u64) -> CorrectionHistoryEntry {
        let index = (key % CORRECTION_HISTORY_SIZE) as usize;
        CorrectionHistoryEntry::from(&self.table[index])
    }

    pub fn set(&self, key: u64, e: CorrectionHistoryEntry) {
        let index = (key % CORRECTION_HISTORY_SIZE) as usize;
        self.table[index].0.store(u64::from(e), Ordering::Relaxed);
    }

    // pub fn add(&mut self, key: u32, e: CorrectionHistoryEntry) {
    //     let index = key as usize % CORRECTION_HISTORY_SIZE as usize;
    //     let bonus = e.value.clamp(-CORRECTION_HISTORY_LIMIT, CORRECTION_HISTORY_LIMIT);
    //     self.table[index].value += bonus - self.table[index].value.abs() / CORRECTION_HISTORY_LIMIT;
    // }

    pub fn clear(&mut self) {
        for entry in self.table.iter_mut() {
            entry.0.store(0, Ordering::Relaxed);
        }
    }
}
