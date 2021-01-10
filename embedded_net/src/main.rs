#![no_std]
#![no_main]

mod serial;

use gd32vf103xx_hal as hal;
use hal::{gpio::GpioExt, pac, prelude::*, rcu::RcuExt};
use longan_nano::led::{rgb, Led};
use panic_halt as _;
use serial::{write_byte, SerialWrapper, STDOUT};
use sparse_embedded::*;
const N_CONV1: usize = 68;
const N_CONV2: usize = 189;
const N_CONV3: usize = 259;
const N_FC1: usize = 177;
const N_FC2: usize = 103;

// const IM: ([[FixedI16; 28]; 28], u8) = include!("../build/im.rs");
// LED's on PC13, PA1 and PA2
const LAYERS: (
    ConvLayer<N_CONV1>,
    ConvLayer<N_CONV2>,
    ConvLayer<N_CONV3>,
    FCLayer<N_FC1>,
    FCLayer<N_FC2>,
) = include!("../build/layers.rs");

struct SerialReader<F: FnMut() -> u8, G: FnMut(u8) -> ()> {
    reader: F,
    sender: G,
}

impl<F: FnMut() -> u8, G: FnMut(u8) -> ()> SerialReader<F, G> {
    fn read_image(&mut self) -> [[FixedI16; 28]; 28] {
        let mut array = [[FixedI16::ZERO; 28]; 28];
        while (self.reader)() != 0xA0 {}
        (self.sender)(0xA0);
        for x in 0..28 {
            for y in 0..28 {
                let b = (self.reader)();
                let n = b as i16 * 4;
                array[x][y] = FixedI16 { n }
            }
        }
        array
    }
    fn send_byte(&mut self, b: u8) {
        (self.sender)(b)
    }
}

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

#[riscv_rt::entry]
fn main() -> ! {
    let dp = pac::Peripherals::take().unwrap();

    // Configure clocks
    let mut rcu = dp
        .RCU
        .configure()
        .ext_hf_clock(8.mhz())
        .sysclk(108.mhz())
        .freeze();

    let mut afio = dp.AFIO.constrain(&mut rcu);

    let gpioa = dp.GPIOA.split(&mut rcu);
    let uart = dp.USART0;
    let tx = gpioa.pa9;
    let rx = gpioa.pa10;
    let tx = tx.into_alternate_push_pull();
    let rx = rx.into_floating_input();
    let config = hal::serial::Config {
        baudrate: 115200.bps(),
        parity: hal::serial::Parity::ParityNone,
        stopbits: hal::serial::StopBits::STOP1,
    };
    let serial = hal::serial::Serial::new(uart, (tx, rx), config, &mut afio, &mut rcu);
    let (tx, mut rx) = serial.split();
    rx.listen();
    let gpioc = dp.GPIOC.split(&mut rcu);
    let (mut red, mut green, mut blue) = rgb(gpioc.pc13, gpioa.pa1, gpioa.pa2);
    let read_byte = || nb::block!(rx.read()).unwrap();
    let send_byte = |b| write_byte(b);
    let mut reader = SerialReader {
        reader: read_byte,
        sender: send_byte,
    };
    riscv::interrupt::free(|_| unsafe {
        STDOUT.replace(SerialWrapper(tx));
    });
    loop {
        red.on();
        green.off();
        blue.off();
        let im = reader.read_image();
        red.off();
        green.off();
        blue.on();
        let res = forward_net(im);
        reader.send_byte(res);
    }
}
