use core::fmt::Debug;
use core::ops::{Add, Mul};
pub mod axis_op;
pub mod conv_unary;
pub mod konst;
pub mod map;
pub mod matmul;
pub mod maxpool;
pub use conv_unary::ConvUnary;
pub use konst::{Const2, Const4};
pub use map::{ReLu2, ReLu4};
pub use matmul::MatMul;
pub use maxpool::MaxPool;
pub trait Datum:
    Debug + Copy + Default + PartialOrd + Add<Output = Self> + Mul<Output = Self>
{
}
impl<F: Debug + Copy + Default + PartialOrd + Add<Output = Self> + Mul<Output = Self>> Datum for F {}
