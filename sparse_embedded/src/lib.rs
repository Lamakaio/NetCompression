#![no_std]
#![feature(min_const_generics)]

use core::{
    cmp::max,
    ops::{Add, AddAssign, Mul, Sub},
};
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct FixedI8 {
    n: i8,
}
impl Add for FixedI8 {
    type Output = FixedI8;

    fn add(self, rhs: Self) -> Self::Output {
        FixedI8 { n: self.n + rhs.n }
    }
}

impl AddAssign for FixedI8 {
    fn add_assign(&mut self, rhs: Self) {
        self.n += rhs.n;
    }
}

impl Sub for FixedI8 {
    type Output = FixedI8;

    fn sub(self, rhs: Self) -> Self::Output {
        FixedI8 { n: self.n - rhs.n }
    }
}

impl Mul for FixedI8 {
    type Output = FixedI8;
    fn mul(self, rhs: Self) -> Self::Output {
        let m = self.n as i16 * rhs.n as i16;
        FixedI8 { n: (m >> 8) as i8 }
    }
}

impl PartialOrd for FixedI8 {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        i8::partial_cmp(&self.n, &other.n)
    }
}

impl Ord for FixedI8 {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        i8::cmp(&self.n, &other.n)
    }
}

impl FixedI8 {
    const ZERO: Self = FixedI8 { n: 0 };
    fn relu(&mut self) {
        self.n = max(0, self.n);
    }
}
pub struct KernelPoint<const K: usize> {
    value: FixedI8,
    i: u8, //this store x in the first 4 bytes, and y in the next 4 bytes. Supports kernel up to 16*16, should be good.
}

//weight and features are split because they are used in separate places, which makes it more cache-friendly.
pub struct ConvLayer<const N: usize, const KERNEL: usize> {
    weight: [KernelPoint<KERNEL>; N],
    features: [u8; N],
}

//implement the operation conv
impl<const KERNEL: usize, const N: usize> ConvLayer<KERNEL, N> {
    pub fn forward<
        const INPUT_FEATURE: usize,
        const OUTPUT_FEATURE: usize,
        const INPUT_SIZE: usize,
        const OUTPUT_SIZE: usize,
    >(
        &self,
        input: [[[FixedI8; INPUT_SIZE]; INPUT_SIZE]; INPUT_FEATURE],
    ) -> [[[FixedI8; OUTPUT_SIZE]; OUTPUT_SIZE]; OUTPUT_FEATURE] {
        let mut output = [[[FixedI8::ZERO; OUTPUT_SIZE]; OUTPUT_SIZE]; OUTPUT_FEATURE];
        let mut feature_start;
        let mut feature_stop = 0;
        for n_out_feature in 0..OUTPUT_FEATURE {
            //compute the start of the points for the feature
            feature_start = feature_stop;
            while self.features[feature_stop] < n_out_feature as u8 {
                feature_stop += 1;
                //if we reach the end, that means all the following kernels are zero and we can just continue.
                if feature_stop >= N {
                    return output;
                }
            }
            for x in 0..OUTPUT_SIZE {
                for y in 0..OUTPUT_SIZE {
                    //computes the (sparse) convolution
                    for f in feature_start..feature_stop {
                        let kernel_point = &self.weight[f];
                        let ker_x = (kernel_point.i >> 4) as usize;
                        let ker_y = (kernel_point.i & 0x0f) as usize;
                        //for every in_feature
                        for n_in_feature in 0..INPUT_FEATURE {
                            output[n_out_feature][x][y] +=
                                input[n_in_feature][x + ker_x][y + ker_y] * kernel_point.value
                        }
                    }
                }
            }
        }
        output
    }
}

pub fn relu<const FEATURE: usize, const SIZE: usize>(
    input: &mut [[[FixedI8; SIZE]; SIZE]; FEATURE],
) {
    for p in input.iter_mut().flatten().flatten() {
        p.relu()
    }
}

pub fn max_pool<
    const FEATURE: usize,
    const IN_SIZE: usize,
    const OUT_SIZE: usize,
    const KERNEL: usize,
    const STRIDE: usize,
>(
    input: [[[FixedI8; IN_SIZE]; IN_SIZE]; FEATURE],
) -> [[[FixedI8; OUT_SIZE]; OUT_SIZE]; FEATURE] {
    let mut output = [[[FixedI8::ZERO; OUT_SIZE]; OUT_SIZE]; FEATURE];
    for f in 0..FEATURE {
        for x in 0..OUT_SIZE {
            for y in 0..OUT_SIZE {
                output[f][x][y] = input[f][x * STRIDE..x * STRIDE + KERNEL]
                    .iter()
                    .map(|c| &c[y * STRIDE..y * STRIDE + KERNEL])
                    .flatten()
                    .copied()
                    .fold(FixedI8::ZERO, FixedI8::max);
            }
        }
    }
    output
}

pub struct FCPoint {
    value: FixedI8,
    i: u16, //there are probably more than 16 weights.
}
pub struct FCLayer<const IN_SIZE: usize, const OUT_SIZE: usize> {
    weights: [[FCPoint; IN_SIZE]; OUT_SIZE],
}

impl<const IN_SIZE: usize, const OUT_SIZE: usize> FCLayer<IN_SIZE, OUT_SIZE> {
    pub fn forward(self, input: [FixedI8; IN_SIZE]) -> [FixedI8; OUT_SIZE] {
        let mut out = [FixedI8::ZERO; OUT_SIZE];
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.weights[i]
                .iter()
                .fold(FixedI8::ZERO, |prev, fc_point| {
                    prev + input[fc_point.i as usize] * fc_point.value
                })
        }
        out
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
