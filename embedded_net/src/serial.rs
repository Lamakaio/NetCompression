//! Stdout based on the UART hooked up to the debug connector

use core::fmt::{self, Write};
use gd32vf103xx_hal::{pac::USART0, prelude::*, serial::Tx};
use nb::block;
use riscv::interrupt;

pub static mut STDOUT: Option<SerialWrapper> = None;

pub struct SerialWrapper(pub Tx<USART0>);

impl fmt::Write for SerialWrapper {
    fn write_str(&mut self, s: &str) -> fmt::Result {
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

/// Writes string to stdout
pub fn write_str(s: &str) {
    interrupt::free(|_| unsafe {
        if let Some(stdout) = STDOUT.as_mut() {
            let _ = stdout.write_str(s);
        }
    })
}

/// Writes formatted string to stdout
pub fn write_fmt(args: fmt::Arguments) {
    interrupt::free(|_| unsafe {
        if let Some(stdout) = STDOUT.as_mut() {
            let _ = stdout.write_fmt(args);
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
