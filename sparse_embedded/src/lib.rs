#![no_std]
use core::{
    cmp::max,
    ops::{Add, AddAssign, Mul, Sub},
};
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct FixedI16 {
    pub n: i16,
}
impl core::fmt::Debug for FixedI16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.n.fmt(f)
    }
}
impl Add for FixedI16 {
    type Output = FixedI16;

    fn add(self, rhs: Self) -> Self::Output {
        FixedI16 { n: self.n + rhs.n }
    }
}

impl AddAssign for FixedI16 {
    fn add_assign(&mut self, rhs: Self) {
        self.n += rhs.n;
    }
}

impl Sub for FixedI16 {
    type Output = FixedI16;

    fn sub(self, rhs: Self) -> Self::Output {
        FixedI16 { n: self.n - rhs.n }
    }
}

impl Mul for FixedI16 {
    type Output = FixedI16;
    fn mul(self, rhs: Self) -> Self::Output {
        let m = self.n as i32 * rhs.n as i32;
        FixedI16 {
            n: (m / 1024) as i16,
        }
    }
}

impl PartialOrd for FixedI16 {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        i16::partial_cmp(&self.n, &other.n)
    }
}

impl Ord for FixedI16 {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        i16::cmp(&self.n, &other.n)
    }
}

impl FixedI16 {
    pub const ZERO: Self = FixedI16 { n: 0 };
    pub fn relu(&mut self) {
        self.n = max(0, self.n);
    }
}
#[derive(Clone, Copy)]
pub struct KernelPoint {
    pub value: FixedI16,
    pub i: u8,
    pub in_feature: u8,
}

impl core::fmt::Debug for KernelPoint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        (self.value, self.i).fmt(f)
    }
}
//weight and features are split because they are used in separate places, which makes it more cache-friendly.
pub struct ConvLayer<const N: usize> {
    pub weight: [KernelPoint; N],
    pub out_features: [u8; N],
}

//implement the operation conv
impl<const N: usize> ConvLayer<N> {
    pub fn forward<
        const INPUT_FEATURE: usize,
        const OUTPUT_FEATURE: usize,
        const INPUT_SIZE: usize,
        const OUTPUT_SIZE: usize,
        const KERNEL: usize,
    >(
        &self,
        input: [[[FixedI16; INPUT_SIZE]; INPUT_SIZE]; INPUT_FEATURE],
    ) -> [[[FixedI16; OUTPUT_SIZE]; OUTPUT_SIZE]; OUTPUT_FEATURE] {
        let mut output = [[[FixedI16::ZERO; OUTPUT_SIZE]; OUTPUT_SIZE]; OUTPUT_FEATURE];
        let mut feature_start;
        let mut feature_stop = 0;
        for n_out_feature in 0..OUTPUT_FEATURE {
            //compute the start of the points for the feature
            feature_start = feature_stop;
            while feature_stop < N && self.out_features[feature_stop] <= n_out_feature as u8 {
                feature_stop += 1;
            }
            for x in 0..OUTPUT_SIZE {
                for y in 0..OUTPUT_SIZE {
                    //computes the (sparse) convolution
                    for f in feature_start..feature_stop {
                        let kernel_point = &self.weight[f];
                        let ker_x = (kernel_point.i % KERNEL as u8) as usize;
                        let ker_y = (kernel_point.i / KERNEL as u8) as usize;
                        //for every in_feature
                        output[n_out_feature][x][y] += input[kernel_point.in_feature as usize]
                            [x + ker_x][y + ker_y]
                            * kernel_point.value
                    }
                }
            }
        }
        output
    }
}

pub fn relu_conv<const FEATURE: usize, const SIZE: usize>(
    mut input: [[[FixedI16; SIZE]; SIZE]; FEATURE],
) -> [[[FixedI16; SIZE]; SIZE]; FEATURE] {
    for p in input.iter_mut().flatten().flatten() {
        p.relu()
    }
    input
}
pub fn relu_fc<const SIZE: usize>(mut input: [FixedI16; SIZE]) -> [FixedI16; SIZE] {
    for p in input.iter_mut() {
        p.relu()
    }
    input
}

pub fn max_pool<
    const FEATURE: usize,
    const IN_SIZE: usize,
    const OUT_SIZE: usize,
    const KERNEL: usize,
    const STRIDE: usize,
>(
    input: [[[FixedI16; IN_SIZE]; IN_SIZE]; FEATURE],
) -> [[[FixedI16; OUT_SIZE]; OUT_SIZE]; FEATURE] {
    let mut output = [[[FixedI16::ZERO; OUT_SIZE]; OUT_SIZE]; FEATURE];
    for f in 0..FEATURE {
        for y in 0..OUT_SIZE {
            for x in 0..OUT_SIZE {
                output[f][x][y] = input[f][x * STRIDE..x * STRIDE + KERNEL]
                    .iter()
                    .map(|c| &c[y * STRIDE..y * STRIDE + KERNEL])
                    .flatten()
                    .copied()
                    .fold(FixedI16::ZERO, FixedI16::max);
            }
        }
    }
    output
}
#[derive(Clone, Copy)]
pub struct FCPoint {
    pub value: FixedI16,
    pub y: u8,
}
impl core::fmt::Debug for FCPoint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        (self.value, self.y).fmt(f)
    }
}
pub struct FCLayer<const N: usize> {
    pub weights: [FCPoint; N],
    pub x: [u8; N],
}
impl<const N: usize> FCLayer<N> {
    pub fn forward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        input: [FixedI16; IN_SIZE],
    ) -> [FixedI16; OUT_SIZE] {
        let mut out = [FixedI16::ZERO; OUT_SIZE];
        let mut o_start;
        let mut o_end = 0;
        for (i, o) in out.iter_mut().enumerate() {
            o_start = o_end;
            while o_end < N && self.x[o_end] <= i as u8 {
                o_end += 1;
            }
            *o = self.weights[o_start..o_end]
                .iter()
                .fold(FixedI16::ZERO, |prev, fc_point| {
                    prev + input[fc_point.y as usize] * fc_point.value
                })
        }
        out
    }
}
