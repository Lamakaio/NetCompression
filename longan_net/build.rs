use embnet::{generate, Datum, DatumType};
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct FixedI16 {
    pub n: i16,
}
impl core::fmt::Display for FixedI16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "FixedI16 {{n: {}}}", self.n)
    }
}

impl PartialOrd for FixedI16 {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        i16::partial_cmp(&self.n, &other.n)
    }
}
impl Datum for FixedI16 {
    fn name() -> &'static str {
        "FixedI16"
    }

    fn datum_type() -> DatumType {
        DatumType::I16
    }
}
impl From<f32> for FixedI16 {
    fn from(v: f32) -> Self {
        let n = (v * 1024.) as i16;
        FixedI16 { n }
    }
}
fn main() {
    generate::<f32, FixedI16, _>("../net.onnx", "net").unwrap();
}
