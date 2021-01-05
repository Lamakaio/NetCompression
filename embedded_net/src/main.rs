#![no_std]
#![no_main]

use nb::block;
use panic_halt as _;

use embedded_hal::blocking::delay::DelayMs;
use gd32vf103xx_hal as hal;
use gd32vf103xx_hal::delay::McycleDelay;
use gd32vf103xx_hal::gpio::GpioExt;
use gd32vf103xx_hal::rcu::RcuExt;
use hal::{
    pac,
    prelude::{_gd32vf103xx_hal_afio_AfioExt, _gd32vf103xx_hal_time_U32Ext},
};
mod serial;
use embedded_hal::serial::Read;
use longan_nano::led::{rgb, Led};
use riscv::interrupt;
use riscv_rt::entry;
use serial::{SerialWrapper, STDOUT};
use sparse_embedded::*;
const N_CONV1: usize = 87;
const N_CONV2: usize = 208;
const N_CONV3: usize = 180;
const N_FC1: usize = 184;
const N_FC2: usize = 181;
// const IM: ([[FixedI16; 28]; 28], u8) = include!("../build/im.rs");
// LED's on PC13, PA1 and PA2
const LAYERS: (
    ConvLayer<N_CONV1>,
    ConvLayer<N_CONV2>,
    ConvLayer<N_CONV3>,
    FCLayer<N_FC1>,
    FCLayer<N_FC2>,
) = include!("../build/layers.rs");

struct SerialReader<F: FnMut() -> u8> {
    reader: F,
}

impl<F: FnMut() -> u8> SerialReader<F> {
    fn read_image(&mut self) -> [[FixedI16; 28]; 28] {
        let mut array = [[FixedI16::ZERO; 28]; 28];
        while (self.reader)() != 0xff {
            sprintln!("waiting for ff");
        }
        sprintln!("finished waiting");
        for x in 0..28 {
            for y in 0..28 {
                let b1 = (self.reader)();
                let b2 = (self.reader)();
                let sign = (self.reader)();
                let n = b1 as i16 + b2 as i16 * 256;
                array[x][y] = FixedI16 {
                    n: if sign == 0 {
                        n
                    } else if sign == 1 {
                        -n
                    } else {
                        sprintln!("Error while receiving");
                        return self.read_image();
                    },
                }
            }
        }
        array
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

#[entry]
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
        baudrate: 115_200.bps(),
        parity: hal::serial::Parity::ParityNone,
        stopbits: hal::serial::StopBits::STOP1,
    };
    let serial = hal::serial::Serial::new(uart, (tx, rx), config, &mut afio, &mut rcu);
    let (tx, mut rx) = serial.split();
    rx.listen();
    let gpioc = dp.GPIOC.split(&mut rcu);
    let (mut red, mut green, mut blue) = rgb(gpioc.pc13, gpioa.pa1, gpioa.pa2);
    // loop {
    //     green.off();
    //     red.on();
    //     sprintln!("received {}", block!(rx.read()).unwrap());
    //     green.on();
    //     red.off();
    //     sprintln!("received {}", block!(rx.read()).unwrap());
    // }
    let read_byte = || {
        sprintln!("reading");
        let b = block!(rx.read()).unwrap_or({
            sprintln!("failed");
            0
        });
        sprintln!("Received {}", b);
        b
    };
    let mut reader = SerialReader { reader: read_byte };
    interrupt::free(|_| unsafe {
        STDOUT.replace(SerialWrapper(tx));
    });
    red.on();
    green.on();
    blue.on();
    sprintln!("Waiting for image");
    let im = reader.read_image();
    red.off();
    green.off();
    sprintln!("Started computing");
    let res = forward_net(im);
    blue.off();
    green.on();
    sprintln!("Computed {}", res);
    let mut delay = McycleDelay::new(&rcu.clocks);
    loop {
        delay.delay_ms(500);
    }
}
