use core::marker::PhantomData;

use super::Datum;
pub struct ReLu4<DATUM: Datum, const N: usize, const C: usize, const H: usize, const W: usize> {
    _casper: PhantomData<DATUM>,
}
impl<DATUM: Datum, const N: usize, const C: usize, const H: usize, const W: usize>
    ReLu4<DATUM, N, C, H, W>
{
    pub const DEFAULT: Self = Self {
        _casper: PhantomData,
    };
    pub fn eval(&self, mut input: [[[[DATUM; W]; H]; C]; N]) -> [[[[DATUM; W]; H]; C]; N] {
        for p in input.iter_mut().flatten().flatten().flatten() {
            if DATUM::default() > *p {
                *p = DATUM::default()
            }
        }

        input
    }
}
pub struct ReLu2<DATUM: Datum, const H: usize, const W: usize> {
    _casper: PhantomData<DATUM>,
}
impl<DATUM: Datum, const H: usize, const W: usize> ReLu2<DATUM, H, W> {
    pub const DEFAULT: Self = Self {
        _casper: PhantomData,
    };
    pub fn eval(&self, mut input: [[DATUM; W]; H]) -> [[DATUM; W]; H] {
        for p in input.iter_mut().flatten() {
            if DATUM::default() > *p {
                *p = DATUM::default()
            }
        }
        input
    }
}
