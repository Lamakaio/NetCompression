#![no_std]
#![no_main]

mod serial;

use core::ops::{Add, Mul};

use embnet_macros::net;
use gd32vf103xx_hal as hal;
use hal::{delay::McycleDelay, gpio::GpioExt, pac, prelude::*, rcu::RcuExt};
use longan_nano::led::{rgb, Led};
use panic_halt as _;
use serial::{write_byte, SerialReader, SerialWrapper, STDOUT};

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct FixedI16 {
    pub n: i16,
}
impl core::fmt::Display for FixedI16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "FixedI16 {{n: {}}}", self.n)
    }
}
impl Add for FixedI16 {
    type Output = FixedI16;

    fn add(self, rhs: Self) -> Self::Output {
        FixedI16 { n: self.n + rhs.n }
    }
}

impl Mul for FixedI16 {
    type Output = FixedI16;
    fn mul(self, rhs: Self) -> Self::Output {
        let m = self.n as i32 * rhs.n as i32;
        FixedI16 {
            n: (m / 1024) as i16,
        }
    }
}

impl PartialOrd for FixedI16 {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        i16::partial_cmp(&self.n, &other.n)
    }
}

#[net(net)]
struct Net {}

fn forward_net(input: [[FixedI16; 28]; 28]) -> u8 {
    let x = Net::NET.eval([[input]]);
    x.0[0]
        .iter()
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
    let net = Net::NET;
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
    let mut delay = McycleDelay::new(&rcu.clocks);
    loop {
        red.on();
        green.on();
        blue.off();
        let im = reader.read_image();
        red.off();
        green.off();
        blue.on();
        let res = forward_net(im);
        reader.send_byte(res);
    }
}
