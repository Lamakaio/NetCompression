fn main() {
    embnet::generate::<f32, _>("../net.onnx", "net").unwrap();
}
