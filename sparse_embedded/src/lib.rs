#![no_std]
#![feature(min_const_generics)]

use core::ops::{Add, Mul, Sub};
#[derive(Clone, Copy)]
pub struct FixedI8 {
    n: i8,
}
impl Add for FixedI8 {
    type Output = FixedI8;

    fn add(self, rhs: Self) -> Self::Output {
        FixedI8 { n: self.n + rhs.n }
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
pub struct KernelPoint<const K: usize> {
    value: FixedI8,
    i: u8, //this store x in the first 4 bytes, and y in the next 4 bytes. Supports kernel up to 16*16, should be good.
}

//weight and features are split because they are used in separate places, which makes it more cache-friendly.
pub struct ConvLayer<const N: usize, const KERNEL: usize> {
    weight: [KernelPoint<KERNEL>; N],
    features: [u8; N],
}

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
        let feature_multi = OUTPUT_FEATURE / INPUT_FEATURE;
        let mut output = [[[FixedI8 { n: 0 }; OUTPUT_SIZE]; OUTPUT_SIZE]; OUTPUT_FEATURE];
        let mut feature_start;
        let mut feature_stop = 0;
        for n_in_feature in 0..INPUT_FEATURE {
            for n_out_feature in n_in_feature * feature_multi..(n_in_feature + 1) * feature_multi {
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
                            output[n_out_feature][x][y] =
                                input[n_in_feature][x + ker_x][y + ker_y] * kernel_point.value;
                        }
                    }
                }
            }
        }
        output
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
