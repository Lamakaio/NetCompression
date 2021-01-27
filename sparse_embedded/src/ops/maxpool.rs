use core::marker::PhantomData;

use super::Datum;
//TODO: this can be simplified a lot when full const generics become stable
//TODO: add support for padding and dilatations.
pub struct MaxPool<
    DATUM: Datum,
    const N: usize,
    const C: usize,
    const H: usize,
    const HO: usize,
    const H_STRIDE: usize,
    const H_KERNEL: usize,
    const W: usize,
    const WO: usize,
    const W_STRIDE: usize,
    const W_KERNEL: usize,
> {
    _casper: PhantomData<*const DATUM>,
}
impl<
        DATUM: Datum,
        const N: usize,
        const C: usize,
        const H: usize,
        const HO: usize,
        const H_STRIDE: usize,
        const H_KERNEL: usize,
        const W: usize,
        const WO: usize,
        const W_STRIDE: usize,
        const W_KERNEL: usize,
    > MaxPool<DATUM, N, C, H, HO, H_STRIDE, H_KERNEL, W, WO, W_STRIDE, W_KERNEL>
{
    pub const DEFAULT: Self = Self {
        _casper: PhantomData,
    };
    pub fn eval(&self, input: &[[[[DATUM; W]; H]; C]; N]) -> [[[[DATUM; WO]; HO]; C]; N] {
        let mut output = [[[[DATUM::default(); WO]; HO]; C]; N];
        for n in 0..N {
            for c in 0..C {
                for h in 0..(H / H_STRIDE) {
                    let h_s = h * H_STRIDE;
                    for w in 0..(W / W_STRIDE) {
                        let w_s = w * W_STRIDE;
                        output[n][c][h][w] = input[n][c][h_s..h_s + H_KERNEL]
                            .iter()
                            .map(|w| w[w_s..w_s + W_KERNEL].iter())
                            .flatten()
                            .copied()
                            .fold(DATUM::default(), |prev, other| {
                                let ret = if prev > other { prev } else { other };
                                ret
                            });
                    }
                }
            }
        }
        output
    }
}
