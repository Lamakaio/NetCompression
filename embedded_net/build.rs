//number of non-zero weights in each layer is hard coded.
const N_CONV1: usize = 87;
const N_CONV2: usize = 208;
const N_CONV3: usize = 180;
const N_FC1: usize = 184;
const N_FC2: usize = 181;
use mnist::{Mnist, MnistBuilder};
use sparse_embedded::*;
use std::io::Write;

fn f64_to_fixed16(i: f64) -> FixedI16 {
    FixedI16 {
        n: (i * 1024.).trunc() as i16,
    }
}

fn main() {
    let Mnist {
        tst_img, tst_lbl, ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(0)
        .validation_set_length(0)
        .test_set_length(1000)
        .finalize();
    let mut file = std::fs::File::create("build/im.rs").unwrap();
    let i = 1;
    file.write_all(&"([[".to_string().into_bytes()).unwrap();
    for x in 0..28 {
        if x > 0 {
            file.write_all(&"],[".to_string().into_bytes()).unwrap();
        }
        for y in 0..28 {
            let v = tst_img[28 * 28 * i + x + 28 * y];
            file.write_all(
                &format!("FixedI16 {{n : {}}},", f64_to_fixed16(v as f64 / 255.).n,).into_bytes(),
            )
            .unwrap();
        }
    }
    file.write_all(&format!("]],{})", tst_lbl[i]).into_bytes())
        .unwrap();
    let conv1_weights = include_bytes!("training_data/conv1_weights.pkl");
    let conv2_weights = include_bytes!("training_data/conv2_weights.pkl");
    let conv3_weights = include_bytes!("training_data/conv3_weights.pkl");
    let fc1_weights = include_bytes!("training_data/fc1_weights.pkl");
    let fc2_weights = include_bytes!("training_data/fc2_weights.pkl");
    let conv1_weights: Vec<Vec<Vec<Vec<f64>>>> = serde_pickle::from_slice(conv1_weights).unwrap();
    let conv2_weights: Vec<Vec<Vec<Vec<f64>>>> = serde_pickle::from_slice(conv2_weights).unwrap();
    let conv3_weights: Vec<Vec<Vec<Vec<f64>>>> = serde_pickle::from_slice(conv3_weights).unwrap();
    let fc1_weights: Vec<Vec<f64>> = serde_pickle::from_slice(fc1_weights).unwrap();
    let fc2_weights: Vec<Vec<f64>> = serde_pickle::from_slice(fc2_weights).unwrap();
    let mut conv1 = ConvLayer {
        weight: [KernelPoint {
            value: FixedI16::ZERO,
            i: 0,
            in_feature: 0,
        }; N_CONV1],
        out_features: [0; N_CONV1],
    };
    let mut conv2 = ConvLayer {
        weight: [KernelPoint {
            value: FixedI16::ZERO,
            i: 0,
            in_feature: 0,
        }; N_CONV2],
        out_features: [0; N_CONV2],
    };
    let mut conv3 = ConvLayer {
        weight: [KernelPoint {
            value: FixedI16::ZERO,
            i: 0,
            in_feature: 0,
        }; N_CONV3],
        out_features: [0; N_CONV3],
    };
    let mut fc1 = FCLayer {
        weights: [FCPoint {
            value: FixedI16::ZERO,
            y: 0,
        }; N_FC1],
        x: [0; N_FC1],
    };
    let mut fc2 = FCLayer {
        weights: [FCPoint {
            value: FixedI16::ZERO,
            y: 0,
        }; N_FC2],
        x: [0; N_FC2],
    };
    //fill in the structures
    let mut c = 0;
    for (out_f, v) in conv1_weights.into_iter().enumerate() {
        for (in_f, v) in v.into_iter().enumerate() {
            for (i, val) in v
                .into_iter()
                .flatten()
                .enumerate()
                .filter(|(_, v)| v.abs() > 0.01)
            {
                conv1.weight[c] = KernelPoint {
                    value: f64_to_fixed16(val),
                    i: i as u8,
                    in_feature: in_f as u8,
                };
                conv1.out_features[c] = out_f as u8;
                c += 1;
            }
        }
    }
    let mut c = 0;
    for (out_f, v) in conv2_weights.into_iter().enumerate() {
        for (in_f, v) in v.into_iter().enumerate() {
            for (i, val) in v
                .into_iter()
                .flatten()
                .enumerate()
                .filter(|(_, v)| v.abs() > 0.01)
            {
                conv2.weight[c] = KernelPoint {
                    value: f64_to_fixed16(val),
                    i: i as u8,
                    in_feature: in_f as u8,
                };
                conv2.out_features[c] = out_f as u8;
                c += 1;
            }
        }
    }
    let mut c = 0;
    for (out_f, v) in conv3_weights.into_iter().enumerate() {
        for (in_f, v) in v.into_iter().enumerate() {
            for (i, val) in v
                .into_iter()
                .flatten()
                .enumerate()
                .filter(|(_, v)| v.abs() > 0.01)
            {
                conv3.weight[c] = KernelPoint {
                    value: f64_to_fixed16(val),
                    i: i as u8,
                    in_feature: in_f as u8,
                };
                conv3.out_features[c] = out_f as u8;
                c += 1;
            }
        }
    }
    let mut c = 0;
    for (x, v) in fc1_weights.into_iter().enumerate() {
        for (y, val) in v.into_iter().enumerate().filter(|(_, v)| v.abs() > 0.01) {
            fc1.weights[c] = FCPoint {
                value: f64_to_fixed16(val),
                y: y as u8,
            };
            fc1.x[c] = x as u8;
            c += 1;
        }
    }
    let mut c = 0;
    for (x, v) in fc2_weights.into_iter().enumerate() {
        for (y, val) in v.into_iter().enumerate().filter(|(_, v)| v.abs() > 0.01) {
            fc2.weights[c] = FCPoint {
                value: f64_to_fixed16(val),
                y: y as u8,
            };
            fc2.x[c] = x as u8;
            c += 1;
        }
    }
    //write the structure in rust to file (yeah this is very inefficient but whatever)
    let mut file = std::fs::File::create("build/layers.rs").unwrap();
    file.write_all(b"( ConvLayer { weight: [").unwrap();
    for k in conv1.weight.iter() {
        file.write_all(
            &format!(
                "KernelPoint {{
            value: FixedI16 {{n: {}}},
            i: {},
            in_feature: {},
        }},",
                k.value.n, k.i, k.in_feature
            )
            .into_bytes(),
        )
        .unwrap();
    }
    file.write_all(b"], out_features : [").unwrap();
    for k in conv1.out_features.iter() {
        file.write_all(&format!("{},", k).into_bytes()).unwrap();
    }
    file.write_all(b"]}, ConvLayer { weight: [").unwrap();
    for k in conv2.weight.iter() {
        file.write_all(
            &format!(
                "KernelPoint {{
            value: FixedI16 {{n: {}}},
            i: {},
            in_feature: {},
        }},",
                k.value.n, k.i, k.in_feature
            )
            .into_bytes(),
        )
        .unwrap();
    }
    file.write_all(b"], out_features : [").unwrap();
    for k in conv2.out_features.iter() {
        file.write_all(&format!("{},", k).into_bytes()).unwrap();
    }
    file.write_all(b"]}, ConvLayer { weight: [").unwrap();
    for k in conv3.weight.iter() {
        file.write_all(
            &format!(
                "KernelPoint {{
            value: FixedI16 {{n: {}}},
            i: {},
            in_feature: {},
        }},",
                k.value.n, k.i, k.in_feature
            )
            .into_bytes(),
        )
        .unwrap();
    }
    file.write_all(b"], out_features : [").unwrap();
    for k in conv3.out_features.iter() {
        file.write_all(&format!("{},", k).into_bytes()).unwrap();
    }
    file.write_all(b"]}, FCLayer {weights: [").unwrap();
    for k in fc1.weights.iter() {
        file.write_all(
            &format!(
                "FCPoint {{
            value: FixedI16 {{n: {}}},
            y: {},
        }},",
                k.value.n, k.y
            )
            .into_bytes(),
        )
        .unwrap();
    }
    file.write_all(b"], x : [").unwrap();
    for k in fc1.x.iter() {
        file.write_all(&format!("{},", k).into_bytes()).unwrap();
    }
    file.write_all(b"]}, FCLayer {weights: [").unwrap();
    for k in fc2.weights.iter() {
        file.write_all(
            &format!(
                "FCPoint {{
            value: FixedI16 {{n: {}}},
            y: {}
        }},",
                k.value.n, k.y
            )
            .into_bytes(),
        )
        .unwrap();
    }
    file.write_all(b"], x : [").unwrap();
    for k in fc2.x.iter() {
        file.write_all(&format!("{},", k).into_bytes()).unwrap();
    }
    file.write_all(b"]})").unwrap();
}
