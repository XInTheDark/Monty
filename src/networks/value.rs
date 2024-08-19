use std::io::Write;
use std::time::Instant;
use crate::{boxed_and_zeroed, Board};

use super::{activation::SCReLU, layer::Layer, QA};

// DO NOT MOVE
#[allow(non_upper_case_globals)]
pub const ValueFileDefaultName: &str = "nn-c3e7b78c4f09.network";

const SCALE: i32 = 400;

#[repr(C)]
pub struct ValueNetwork {
    l1: Layer<i16, { 768 * 4 }, 2048>,
    l2: Layer<f32, 2048, 16>,
    l3: Layer<f32, 16, 16>,
    l4: Layer<f32, 16, 16>,
    l5: Layer<f32, 16, 16>,
    l6: Layer<f32, 16, 16>,
    l7: Layer<f32, 16, 16>,
    l8: Layer<f32, 16, 16>,
    l9: Layer<f32, 16, 16>,
    l10: Layer<f32, 16, 16>,
    l11: Layer<f32, 16, 1>,
}

impl ValueNetwork {
    // log time taken for each layer
    pub fn eval(&self, board: &Board) -> i32 {
        let mut str = String::new();
        let mut time = Instant::now();
        let time2 = Instant::now();
        let l2 = self.l1.forward(board);
        str.push_str(&format!("l1: {}\n", time.elapsed().as_micros())); time = Instant::now();
        let l3 = self.l2.forward_from_i16::<SCReLU>(&l2);
        str.push_str(&format!("l2: {}\n", time.elapsed().as_micros())); time = Instant::now();
        let l4 = self.l3.forward::<SCReLU>(&l3);
        str.push_str(&format!("l3: {}\n", time.elapsed().as_micros())); time = Instant::now();
        let l5 = self.l4.forward::<SCReLU>(&l4);
        str.push_str(&format!("l4: {}\n", time.elapsed().as_micros())); time = Instant::now();
        let l6 = self.l5.forward::<SCReLU>(&l5);
        str.push_str(&format!("l5: {}\n", time.elapsed().as_micros())); time = Instant::now();
        let l7 = self.l6.forward::<SCReLU>(&l6);
        str.push_str(&format!("l6: {}\n", time.elapsed().as_micros())); time = Instant::now();
        let l8 = self.l7.forward::<SCReLU>(&l7);
        str.push_str(&format!("l7: {}\n", time.elapsed().as_micros())); time = Instant::now();
        let l9 = self.l8.forward::<SCReLU>(&l8);
        str.push_str(&format!("l8: {}\n", time.elapsed().as_micros())); time = Instant::now();
        let l10 = self.l9.forward::<SCReLU>(&l9);
        str.push_str(&format!("l9: {}\n", time.elapsed().as_micros())); time = Instant::now();
        let l11 = self.l10.forward::<SCReLU>(&l10);
        str.push_str(&format!("l10: {}\n", time.elapsed().as_micros())); time = Instant::now();
        let out = self.l11.forward::<SCReLU>(&l11);
        str.push_str(&format!("l11: {}\n", time.elapsed().as_micros())); time = Instant::now();

        let time2 = time2.elapsed().as_micros();
        if time2 > 1000 {
            str = format!("Time: {}\n", time2) + &str + "\n";

            // write to file debug3.txt

            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .append(true)
                .open("debug3.txt")
                .unwrap();

            writeln!(file, "{}", str).unwrap();

        }

        (out.0[0] * SCALE as f32) as i32
    }
}

#[repr(C)]
pub struct UnquantisedValueNetwork {
    l1: Layer<f32, { 768 * 4 }, 2048>,
    l2: Layer<f32, 2048, 16>,
    l3: Layer<f32, 16, 16>,
    l4: Layer<f32, 16, 16>,
    l5: Layer<f32, 16, 16>,
    l6: Layer<f32, 16, 16>,
    l7: Layer<f32, 16, 16>,
    l8: Layer<f32, 16, 16>,
    l9: Layer<f32, 16, 16>,
    l10: Layer<f32, 16, 16>,
    l11: Layer<f32, 16, 1>,
}

impl UnquantisedValueNetwork {
    pub fn quantise(&self) -> Box<ValueNetwork> {
        let mut quantised: Box<ValueNetwork> = unsafe { boxed_and_zeroed() };

        self.l1.quantise_into(&mut quantised.l1, QA);

        quantised.l2 = self.l2;
        quantised.l3 = self.l3;
        quantised.l4 = self.l4;
        quantised.l5 = self.l5;
        quantised.l6 = self.l6;
        quantised.l7 = self.l7;
        quantised.l8 = self.l8;
        quantised.l9 = self.l9;
        quantised.l10 = self.l10;
        quantised.l11 = self.l11;

        quantised
    }
}
