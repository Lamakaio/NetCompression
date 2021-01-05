use mnist::{Mnist, MnistBuilder};
use sparse_embedded::*;
const N_CONV1: usize = 87;
const N_CONV2: usize = 208;
const N_CONV3: usize = 180;
const N_FC1: usize = 184;
const N_FC2: usize = 181;
const IM: ([[FixedI16; 28]; 28], u8) = include!("../../embedded_net/build/im.rs");
const LAYERS: (
    ConvLayer<N_CONV1>,
    ConvLayer<N_CONV2>,
    ConvLayer<N_CONV3>,
    FCLayer<N_FC1>,
    FCLayer<N_FC2>,
) = include!("../../embedded_net/build/layers.rs");

fn forward_net(input: [[FixedI16; 28]; 28]) -> u8 {
    let (conv1, conv2, conv3, fc1, fc2) = &LAYERS;
    let x = max_pool::<6, 24, 12, 2, 2>(relu_conv(conv1.forward::<1, 6, 28, 24, 5>([input])));

    let x = max_pool::<16, 8, 4, 2, 2>(relu_conv(conv2.forward::<6, 16, 12, 8, 5>(x)));
    let x = relu_conv(conv3.forward::<16, 120, 4, 1, 4>(x));
    let mut fc_input = [FixedI16::ZERO; 120];
    for (i, v) in x.iter().enumerate() {
        fc_input[i] = v[0][0]
    }
    let x = relu_fc(fc1.forward::<120, 84>(fc_input));
    let x = fc2.forward::<84, 10>(x);
    x.iter()
        .copied()
        .enumerate()
        .fold(None, |p, v| {
            if p.is_none() {
                Some(v)
            } else {
                if v.1 > p.unwrap().1 {
                    Some(v)
                } else {
                    p
                }
            }
        })
        .unwrap()
        .0 as u8
}

fn f64_to_fixed16(i: f64) -> FixedI16 {
    FixedI16 {
        n: (i * 1024.).trunc() as i16,
    }
}

fn main() {
    let (trn_size, rows, cols) = (50_000, 28, 28);

    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(0)
        .validation_set_length(0)
        .test_set_length(1_000)
        .finalize();
    let mut error = 0;
    let res = forward_net(IM.0);
    println!("{}, {}", res, IM.1);
    let i = 1;
    let mut im = [[FixedI16::ZERO; 28]; 28];
    for (j, v) in tst_img[28 * 28 * i..28 * 28 * (i + 1)].iter().enumerate() {
        im[j % 28][j / 28] = f64_to_fixed16(*v as f64 / 255.);
    }
    println!("{:?}\n\n\n\n{:?}", IM.0[10], im[10]);
    println!("{}, {}", forward_net(im), tst_lbl[i]);
    // for i in 0..1000 {
    //     let mut im = [[FixedI16::ZERO; 28]; 28];
    //     for (j, v) in tst_img[28 * 28 * i..28 * 28 * (i + 1)].iter().enumerate() {
    //         im[j % 28][j / 28] = f64_to_fixed16(*v as f64 / 255.);
    //     }
    //     if forward_net(im) != tst_lbl[i] {
    //         error += 1
    //     }
    // }
    // println!("{}", error as f32 / 10000.);
}
