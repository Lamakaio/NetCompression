#![feature(global_asm)]
#![no_main]
#![no_std]
#![allow(unused_parens)]
#![allow(non_upper_case_globals)]
use sparse_embedded::*;
const N_CONV1: usize = 87;
const N_CONV2: usize = 208;
const N_CONV3: usize = 180;
const N_FC1: usize = 184;
const N_FC2: usize = 181;

// LED's on PC13, PA1 and PA2
// We will use PA1 (green only)
const rcu_apb2en: u32 = (0x4002_1000 + 0x18);
const LAYERS: (
    ConvLayer<N_CONV1>,
    ConvLayer<N_CONV2>,
    ConvLayer<N_CONV3>,
    FCLayer<N_FC1>,
    FCLayer<N_FC2>,
) = include!("../build/layers.rs");
const gpioa_ctl0: u32 = (0x4001_0800 + 0x0);
const gpioa_data: u32 = (0x4001_0800 + 0xc);

extern crate panic_abort;

// The reset handler
#[no_mangle]
pub unsafe extern "C" fn Reset() -> ! {
    r0::zero_bss(&mut _sbss, &mut _ebss);
    r0::init_data(&mut _sdata, &mut _edata, &_sidata);
    main()
}

fn init_ports() {
    unsafe {
        // Enable clock to Port A and Port C
        let x = core::ptr::read_volatile(rcu_apb2en as *mut u32);
        core::ptr::write_volatile(rcu_apb2en as *mut u32, x | (1 << 2));
        // Enable push-pull o/p Port A, pins 1 and 2.
        let x = core::ptr::read_volatile(gpioa_ctl0 as *mut u32);
        core::ptr::write_volatile(gpioa_ctl0 as *mut u32, x | (1 << 4));
    }
}

// don't compile with optimization enabled!
// fn delay(mut n: u32) {
//     while n != 0 {
//         unsafe {
//             write_volatile(&mut n, n - 1);
//         }
//     }
// }

// // Blink Green LED (PA1).
// fn blink_led() {
//     let mut bits: u32 = !(1 << 1);
//     loop {
//         unsafe {
//             // LED on when PA1 bit is 0
//             core::ptr::write_volatile(gpioa_data as *mut u32, bits);
//         }
//         delay(0x4ffff);
//         bits = !bits;
//     }
// }

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
                if v.0 > p.unwrap().0 {
                    Some(v)
                } else {
                    p
                }
            }
        })
        .unwrap()
        .0 as u8
}

fn main() -> ! {
    init_ports();
    // blink_led();
    let mut input = [[FixedI16::ZERO; 28]; 28];
    for x in 0..28 {
        for y in 0..28 {
            unsafe {
                input[x][y] = core::ptr::read_volatile(gpioa_ctl0 as *mut FixedI16);
            }
        }
    }
    let x = forward_net(input);
    unsafe {
        core::ptr::write_volatile(gpioa_data as *mut u8, x);
    }
    loop {}
}

extern "C" {
    // Boundaries of the .bss section
    static mut _ebss: u32;
    static mut _sbss: u32;

    // Boundaries of the .data section
    static mut _edata: u32;
    static mut _sdata: u32;

    // Initial values of the .data section (stored in Flash)
    static _sidata: u32;
}

// Make sure there is an abort when linking
#[cfg(target_arch = "riscv32")]
global_asm!(
    r#"
lui sp, %hi(__stacktop)
call Reset
.globl abort
abort:
  jal zero, abort
"#
);
