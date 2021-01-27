use std::{io, thread::sleep, time::Duration};

use colour::{green_ln, red_ln};
use minifb::{Key, Window, WindowOptions};
use mnist::{Mnist, MnistBuilder};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serial::SerialPort;
use text_io::read;

const SCALE: usize = 6;
fn send_image<T: SerialPort>(port: &mut T, im: [[u8; 28]; 28]) -> io::Result<u8> {
    let mut buf = [0xA0; 1];
    port.write(&buf)?;
    port.read(&mut buf)?;
    while buf[0] != 0xA0 {
        port.write(&buf)?;
        port.read(&mut buf)?;
    }
    for x in 0..28 {
        for y in 0..28 {
            port.write(&im[y][x..x + 1])?;
        }
    }
    port.read(&mut buf)?;
    //if there was an error, retry
    if buf[0] == 0xE0 {
        port.write(&buf)?;
        return send_image(port, im);
    }
    Ok(buf[0])
}

fn configure<T: SerialPort>(port: &mut T) -> io::Result<()> {
    port.reconfigure(&|settings| {
        settings.set_baud_rate(serial::Baud115200)?;
        settings.set_char_size(serial::Bits8);
        settings.set_parity(serial::ParityNone);
        settings.set_stop_bits(serial::Stop1);
        settings.set_flow_control(serial::FlowNone);
        Ok(())
    })?;
    port.set_timeout(Duration::from_millis(3000))?;
    Ok(())
}
fn main() {
    let Mnist {
        tst_img, tst_lbl, ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(0)
        .validation_set_length(0)
        .test_set_length(10_000)
        .finalize();
    println!("Please input the path to the serial port");
    let port: String = {
        let r: String = read!("{}\n");
        if r.is_empty() {
            "/dev/ttyUSB0".to_string()
        } else {
            r
        }
    };
    println!("Please select a set size. It will randomly be selected from the MNIST test set.");
    let s: u32 = read!("{}\n");
    let mut v: Vec<usize> = (0..10_000usize).collect();
    v.shuffle(&mut thread_rng());
    let im_screen_size = (s as f32).sqrt().ceil() as usize;
    let screen_size = SCALE * 28 * im_screen_size;
    let mut window = Window::new(
        "Images to be sent",
        screen_size,
        screen_size,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // Limit to max ~60 fps update rate
    let mut buffer: Vec<u32> = vec![0; screen_size * screen_size];
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));
    let mut images = Vec::new();
    for i in 0..s as usize {
        let im_i = v[i];
        let mut im = [[0; 28]; 28];
        let x_screen = i % im_screen_size;
        let y_screen = i / im_screen_size;
        for x in 0..28 {
            for y in 0..28 {
                let val = tst_img[28 * 28 * im_i + x + 28 * y];
                for xx in 0..SCALE {
                    for yy in 0..SCALE {
                        buffer[28 * SCALE * (x_screen + y_screen * screen_size)
                            + (x * SCALE + xx)
                            + (SCALE * y + yy) * screen_size] =
                            (val as u32) + (val as u32) * 256 + (val as u32) * 256 * 256;
                    }
                }
                im[x][y] = val;
            }
        }
        images.push(im);
    }

    window
        .update_with_buffer(&buffer, screen_size, screen_size)
        .unwrap();

    //Image transfer
    let mut port = serial::open(&port).unwrap();
    let mut err = 0;
    configure(&mut port).unwrap();
    for (i, im) in images.into_iter().enumerate() {
        let res = send_image(&mut port, im).unwrap();
        if res == tst_lbl[v[i]] {
            green_ln!("Image {}: output is {}, label is {}", i, res, tst_lbl[v[i]]);
        } else {
            err += 1;
            red_ln!("Image {}: output is {}, label is {}", i, res, tst_lbl[v[i]]);
        }
    }
    println!("Error rate : {}", err as f32 / s as f32);
    while window.is_open() && !window.is_key_down(Key::Escape) {
        sleep(Duration::from_millis(10));
        window.update();
    }
}
