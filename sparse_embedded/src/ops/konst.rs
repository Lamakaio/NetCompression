use super::Datum;

#[derive(Default, Clone, Copy)]
pub struct DataPoint2<DATUM: Datum> {
    pub value: DATUM,
    pub h: u8,
    pub w: u8, //TODO: support matrix bigger than 255*255
}

pub struct Const2<DATUM: Datum, const H: usize, const W: usize, const N_POINTS: usize> {
    pub data: [DataPoint2<DATUM>; N_POINTS],
}
impl<DATUM: Datum, const H: usize, const W: usize, const N_POINTS: usize>
    Const2<DATUM, H, W, N_POINTS>
{
    pub fn eval(&self) -> &[DataPoint2<DATUM>; N_POINTS] {
        &self.data
    }
}

#[derive(Default, Clone, Copy)]
pub struct DataPoint4<DATUM: Datum> {
    value: DATUM,
    o: u8, //TODO: support more than 255 channels
    i: u8,
    hw: u8, //TODO: support kernels bigger than 16*16
}
pub struct Const4<
    DATUM: Datum,
    const O: usize,
    const I: usize,
    const H: usize,
    const W: usize,
    const N_POINTS: usize,
> {
    pub data: [DataPoint4<DATUM>; N_POINTS],
}
impl<
        DATUM: Datum,
        const O: usize,
        const I: usize,
        const H: usize,
        const W: usize,
        const N_POINTS: usize,
    > Const4<DATUM, O, I, H, W, N_POINTS>
{
    pub fn eval(&self) -> &[DataPoint4<DATUM>; N_POINTS] {
        &self.data
    }
}
