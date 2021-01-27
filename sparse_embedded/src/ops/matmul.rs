use core::marker::PhantomData;

use super::{konst::DataPoint2, Datum};
//matric multiplication between a sparse matrix 1 and a full matrix 2
pub struct MatMul<
    DATUM: Datum,
    const HI1: usize,
    const WI1: usize,
    const HI2: usize,
    const N_POINTS: usize,
> {
    _casper: PhantomData<DATUM>,
}

impl<DATUM: Datum, const HI1: usize, const WI1: usize, const HI2: usize, const N_POINTS: usize>
    MatMul<DATUM, HI1, WI1, HI2, N_POINTS>
{
    pub const DEFAULT: Self = Self {
        _casper: PhantomData,
    };
    pub fn eval(
        &self,
        input1: &[DataPoint2<DATUM>; N_POINTS],
        input2: &[[DATUM; HI1]; HI2],
    ) -> [[DATUM; WI1]; HI2] {
        let mut output = [[DATUM::default(); WI1]; HI2];
        for data_point in input1.iter() {
            for ho in 0..HI2 {
                output[ho][data_point.w as usize] = output[ho][data_point.w as usize]
                    + input2[ho][data_point.h as usize] * data_point.value;
            }
        }
        output
    }
}
