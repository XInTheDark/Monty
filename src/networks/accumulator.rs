use std::ops::{AddAssign, Mul};
use std::time::Instant;
use std::io::Write;

use super::activation::Activation;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Accumulator<T: Copy, const N: usize>(pub [T; N]);

impl<T: AddAssign<T> + Copy + Mul<T, Output = T>, const N: usize> Accumulator<T, N> {
    pub fn add(&mut self, other: &Self) {
        let time = Instant::now();
        for (i, &j) in self.0.iter_mut().zip(other.0.iter()) {
            *i += j;
        }
        let time2 = time.elapsed().as_micros();
        // log to debug4
        if time2 > 100 {
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .append(true)
                .open("debug4.txt")
                .unwrap();

            writeln!(file, "acc add {}", time2).unwrap();
        }
    }

    pub fn madd(&mut self, mul: T, other: &Self) {
        for (i, &j) in self.0.iter_mut().zip(other.0.iter()) {
            *i += mul * j;
        }
    }
}

impl<const N: usize> Accumulator<f32, N> {
    pub fn dot<T: Activation>(&self, other: &Self) -> f32 {
        let mut res = 0.0;

        for (i, j) in self.0.iter().zip(other.0.iter()) {
            res += T::activate(*i) * T::activate(*j);
        }

        res
    }

    pub fn quantise(&self, qa: i16) -> Accumulator<i16, N> {
        let mut res = Accumulator([0; N]);

        for (i, &j) in res.0.iter_mut().zip(self.0.iter()) {
            if j > 1.98 {
                println!("{j}")
            }

            *i = (j * f32::from(qa)) as i16;
        }

        res
    }
}
