//! Stdout based on the UART hooked up to the debug connector

use super::FixedI16;
use gd32vf103xx_hal::{pac::USART0, prelude::*, serial::Tx};
use nb::block;
use riscv::interrupt;

pub static mut STDOUT: Option<SerialWrapper> = None;

pub struct SerialWrapper(pub Tx<USART0>);

impl core::fmt::Write for SerialWrapper {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for byte in s.as_bytes() {
            if *byte == '\n' as u8 {
                let res = block!(self.0.write('\r' as u8));

                if res.is_err() {
                    return Err(::core::fmt::Error);
                }
            }

            let res = block!(self.0.write(*byte));

            if res.is_err() {
                return Err(::core::fmt::Error);
            }
        }
        Ok(())
    }
}
impl SerialWrapper {
    fn write_byte(&mut self, byte: u8) -> core::fmt::Result {
        let res = block!(self.0.write(byte));

        if res.is_err() {
            return Err(::core::fmt::Error);
        }
        Ok(())
    }
}
pub fn write_byte(b: u8) {
    interrupt::free(|_| unsafe {
        if let Some(stdout) = STDOUT.as_mut() {
            let _ = stdout.write_byte(b);
        }
    })
}

/// Macro for printing to the serial standard output
#[macro_export]
macro_rules! sprint {
    ($s:expr) => {
        $crate::serial::write_str($s)
    };
    ($($tt:tt)*) => {
        $crate::serial::write_fmt(format_args!($($tt)*))
    };
}

/// Macro for printing to the serial standard output, with a newline.
#[macro_export]
macro_rules! sprintln {
    () => {
        $longan::serial::write_str("\n")
    };
    ($s:expr) => {
        $crate::serial::write_str(concat!($s, "\n"))
    };
    ($s:expr, $($tt:tt)*) => {
        $crate::serial::write_fmt(format_args!(concat!($s, "\n"), $($tt)*))
    };
}

pub struct SerialReader<F: FnMut() -> u8, G: FnMut(u8) -> ()> {
    pub reader: F,
    pub sender: G,
}

impl<F: FnMut() -> u8, G: FnMut(u8) -> ()> SerialReader<F, G> {
    pub fn read_image(&mut self) -> [[FixedI16; 28]; 28] {
        let mut array = [[FixedI16::default(); 28]; 28];
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
    pub fn send_byte(&mut self, b: u8) {
        (self.sender)(b)
    }
}
