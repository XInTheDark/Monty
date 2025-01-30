use std::sync::atomic::{AtomicU64, Ordering};
use std::mem::{transmute};

static CORRHIST_SIZE: usize = 1 << 16;
#[derive(Clone, Copy, Debug, Default)]
pub struct CorrHistEntry {
    pub delta_sum: f32,
    pub weight_sum: f32,
}

impl CorrHistEntry {
    #[inline]
    pub fn delta(&self) -> f32 {
        if self.weight_sum.abs() < f32::EPSILON {
            0.0
        } else {
            self.delta_sum / self.weight_sum
        }
    }
}

#[derive(Default)]
struct CorrHistEntryInternal(AtomicU64);

impl Clone for CorrHistEntryInternal {
    fn clone(&self) -> Self {
        Self(AtomicU64::new(self.0.load(Ordering::Relaxed)))
    }
}

impl From<&CorrHistEntryInternal> for CorrHistEntry {
    fn from(value: &CorrHistEntryInternal) -> Self {
        unsafe { transmute(value.0.load(Ordering::Relaxed)) }
    }
}

impl From<CorrHistEntry> for u64 {
    fn from(value: CorrHistEntry) -> Self {
        unsafe { transmute(value) }
    }
}

pub struct CorrHistTable {
    table: Vec<CorrHistEntryInternal>,
}

impl CorrHistTable {
    pub fn new(threads: usize) -> Self {
        let size = CORRHIST_SIZE;
        let chunk_size = size.div_ceil(threads);

        let mut table = CorrHistTable { table: Vec::new() };
        table.table.reserve_exact(size);

        unsafe {
            use std::mem::{size_of, MaybeUninit};
            let ptr = table.table.as_mut_ptr().cast();
            let uninit: &mut [MaybeUninit<u8>] =
                std::slice::from_raw_parts_mut(ptr, size * size_of::<CorrHistEntryInternal>());

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
        let chunk_size = self.table.len().div_ceil(threads);

        std::thread::scope(|s| {
            for chunk in self.table.chunks_mut(chunk_size) {
                s.spawn(|| {
                    for entry in chunk.iter_mut() {
                        *entry = CorrHistEntryInternal::default();
                    }
                });
            }
        });
    }

    pub fn get_or_create(&self, ch_hash: u64) -> CorrHistEntry {
        let idx = ch_hash % (self.table.len() as u64);
        CorrHistEntry::from(&self.table[idx as usize])
    }

    // increment delta and weight
    pub fn update(&self, ch_hash: u64, delta: f32, weight: f32) {
        let idx = (ch_hash % (self.table.len() as u64)) as usize;
        let entry = &self.table[idx];
        loop {
            let old_bits = entry.0.load(Ordering::Relaxed);
            let old_entry: CorrHistEntry = unsafe { transmute(old_bits) };

            let new_entry = CorrHistEntry {
                delta_sum: old_entry.delta_sum + delta,
                weight_sum: old_entry.weight_sum + weight,
            };
            let new_bits: u64 = unsafe { transmute(new_entry) };

            match entry.0.compare_exchange_weak(
                old_bits,
                new_bits,
                Ordering::Relaxed,
                Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }
    }
}