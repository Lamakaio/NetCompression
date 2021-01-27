use core::marker::PhantomData;

use super::Datum;

pub struct RmW<DATUM: Datum, const N: usize, const C: usize, const H: usize> {
    _casper: PhantomData<DATUM>,
}

impl<DATUM: Datum, const N: usize, const C: usize, const H: usize> RmW<DATUM, N, C, H> {
    pub const DEFAULT: Self = Self {
        _casper: PhantomData,
    };
    pub fn eval(&self, input: [[[[DATUM; 1]; H]; C]; N]) -> [[[DATUM; H]; C]; N] {
        let mut output = [[[DATUM::default(); H]; C]; N];
        for n in 0..N {
            for c in 0..C {
                for h in 0..H {
                    output[n][c][h] = input[n][c][h][0]
                }
            }
        }
        output
    }
}
pub struct RmH<DATUM: Datum, const N: usize, const C: usize> {
    _casper: PhantomData<DATUM>,
}

impl<DATUM: Datum, const N: usize, const C: usize> RmH<DATUM, N, C> {
    pub const DEFAULT: Self = Self {
        _casper: PhantomData,
    };
    pub fn eval(&self, input: [[[DATUM; 1]; C]; N]) -> [[DATUM; C]; N] {
        let mut output = [[DATUM::default(); C]; N];
        for n in 0..N {
            for c in 0..C {
                output[n][c] = input[n][c][0]
            }
        }
        output
    }
}
