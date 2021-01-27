use super::Datum;
#[derive(Default, Clone, Copy)]
pub struct KernelPoint<DATUM: Datum> {
    pub value: DATUM,
    pub i: u8,          //TODO: kernel are at most 16*16, add support for bigger kernels
    pub in_feature: u8, //TODO: add support for > 255 features
    pub out_feature: u8,
}

//TODO: this can be simplified a lot when full const generics become stable
//TODO: add support for padding, dilatations, strides.
pub struct ConvUnary<
    DATUM: Datum,
    const N: usize,
    const C: usize,
    const CO: usize,
    const H: usize,
    const HO: usize,
    const H_KERNEL: usize,
    const W: usize,
    const WO: usize,
    const W_KERNEL: usize,
    const N_POINTS: usize,
> {
    pub kernel: [KernelPoint<DATUM>; N_POINTS],
}
impl<
        DATUM: Datum,
        const N: usize,
        const C: usize,
        const CO: usize,
        const H: usize,
        const HO: usize,
        const H_KERNEL: usize,
        const W: usize,
        const WO: usize,
        const W_KERNEL: usize,
        const N_POINTS: usize,
    > ConvUnary<DATUM, N, C, CO, H, HO, H_KERNEL, W, WO, W_KERNEL, N_POINTS>
{
    pub fn eval(&self, input: &[[[[DATUM; W]; H]; C]; N]) -> [[[[DATUM; WO]; HO]; CO]; N] {
        let mut output = [[[[DATUM::default(); WO]; HO]; CO]; N];
        for x in 0..HO {
            for y in 0..WO {
                for kernel_point in self.kernel.iter() {
                    let ker_x = (kernel_point.i % 16) as usize;
                    let ker_y = (kernel_point.i / 16) as usize;
                    for n_batch in 0..N {
                        output[n_batch][kernel_point.out_feature as usize][x][y] = output[n_batch]
                            [kernel_point.out_feature as usize][x][y]
                            + input[n_batch][kernel_point.in_feature as usize][x + ker_x][y + ker_y]
                                * kernel_point.value
                    }
                }
            }
        }
        output
    }
}
