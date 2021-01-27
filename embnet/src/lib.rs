pub use embnet_macros::net;
use std::{
    borrow::{Borrow, Cow},
    collections::HashMap,
    io::{BufWriter, Write},
    path::Path,
    sync::Arc,
    todo, write,
};
pub use tract_core::prelude::{Datum, DatumType};
use tract_core::{
    internal::AxisOp,
    ops::{
        binary::UnaryOp,
        cnn::{ConvUnary, MaxPool},
        konst::Const,
        math::Max,
        matmul::MatMul,
        source::TypedSource,
    },
    prelude::{SymbolValues, Tensor},
    tract_data::TVec,
};
use tract_onnx::{
    onnx,
    prelude::{Framework, InferenceModelExt, TractResult, TypedNode},
};
pub fn generate<InDatum: Datum, OutDatum: Datum + From<InDatum>, P: AsRef<Path>>(
    p: P,
    name: &str,
) -> TractResult<()> {
    //only typed models can be compiled (as all size need to be static)
    let plan = onnx().model_for_path(p)?.into_typed()?.into_runnable()?;
    let prefix = std::env::var("OUT_DIR").unwrap_or(String::new()) + "embnet_build/";
    std::fs::create_dir(&prefix).unwrap_or(()); //ignore errors
    let value_file = std::fs::File::create(prefix.clone() + &*format!("{}_value.rs", name))?;
    let type_file = std::fs::File::create(prefix.clone() + &*format!("{}_type.rs", name))?;
    let impl_file = std::fs::File::create(prefix + &*format!("{}_impl.rs", name))?;
    let mut const_n_points = HashMap::new();
    let mut value_file_buffer = BufWriter::new(value_file);
    let mut type_file_buffer = BufWriter::new(type_file);
    let mut impl_file_buffer = BufWriter::new(impl_file);
    let output_sizes = plan
        .model
        .nodes
        .iter()
        .map(|n| {
            Ok(n.outputs
                .iter()
                //I have to understand in which case there are symbols, and how I can get them if they are necessary
                .map(|o| Ok(o.fact.shape.eval(&SymbolValues::default())?))
                .collect::<TractResult<_>>()?)
        })
        .collect::<TractResult<_>>()?;
    println!("---{:#?}", output_sizes);
    write!(value_file_buffer, "Self {{")?;
    let mut impl_str = String::new();
    let mut input_str = String::new();
    for node_id in plan.order {
        generate_node::<InDatum, OutDatum, _>(
            &plan.model.nodes[node_id],
            &output_sizes,
            &mut const_n_points,
            &mut value_file_buffer,
            &mut type_file_buffer,
            &mut impl_str,
            &mut input_str,
        )?;
    }
    write!(value_file_buffer, "}}")?;
    let output_string = plan
        .outputs
        .iter()
        .map(|outlet_id| {
            let shape = &output_sizes[outlet_id.node][0];
            let mut o_string = String::new();
            for _ in 0..shape.len() {
                o_string.push_str("[");
            }
            o_string.push_str(OutDatum::name());
            let mut v = shape.iter().collect::<Vec<&usize>>();
            v.reverse();
            for s in v {
                o_string.push_str(&*format!("; {}]", s));
            }
            o_string.push_str(",");
            o_string
        })
        .collect::<String>();
    let return_string = plan
        .outputs
        .iter()
        .map(|outlet_id| format!("data_{}, ", outlet_id.node))
        .collect::<String>();
    write!(
        impl_file_buffer,
        "
fn eval(&self, {}) -> ({}) {{
    {}
    ({})
}}",
        input_str, output_string, impl_str, return_string
    )?;
    impl_file_buffer.flush()?;
    value_file_buffer.flush()?;
    type_file_buffer.flush()?;
    Ok(())
}

fn generate_node<InDatum: Datum, OutDatum: Datum + From<InDatum>, W: Write>(
    node: &TypedNode,
    output_sizes: &Vec<Vec<Cow<TVec<usize>>>>,
    const_n_points: &mut HashMap<usize, usize>,
    value_dest: &mut W,
    type_dest: &mut W,
    impl_str: &mut String,
    input_str: &mut String,
) -> TractResult<()> {
    let input_sizes = node
        .inputs
        .iter()
        .map(|outlet_id| output_sizes[outlet_id.node][outlet_id.slot].borrow())
        .collect::<Vec<&TVec<usize>>>();
    let name;
    if let Some(max_pool) = node.op.as_any().downcast_ref::<MaxPool>() {
        name = format!("max_pool{}", node.id);
        generate_max_pool::<InDatum, OutDatum, _>(
            &*input_sizes[0],
            max_pool,
            &*name,
            value_dest,
            type_dest,
        )?;
        impl_str.push_str(&*format!(
            "let data_{} = self.{}.eval(&data_{});\n",
            node.id, name, node.inputs[0].node
        )); //TODO : multi outputs
    } else if let Some(un_op) = node.op.as_any().downcast_ref::<UnaryOp>() {
        name = format!("un_op{}", node.id);
        generate_unary_op::<OutDatum, _>(un_op, &*input_sizes[0], &*name, value_dest, type_dest)?;
        impl_str.push_str(&*format!(
            "let data_{} = self.{}.eval(data_{});\n",
            node.id, name, node.inputs[0].node
        ));
    } else if let Some(conv) = node.op.as_any().downcast_ref::<ConvUnary>() {
        name = format!("conv{}", node.id);
        generate_conv_unary::<InDatum, OutDatum, _>(
            conv,
            &*input_sizes[0],
            &*name,
            value_dest,
            type_dest,
        )?;
        impl_str.push_str(&*format!(
            "let data_{} = self.{}.eval(&data_{});\n",
            node.id, name, node.inputs[0].node
        ));
    } else if let Some(Const(konst)) = node.op.as_any().downcast_ref::<Const>() {
        name = format!("konst{}", node.id);
        let output_shape: &TVec<usize> = output_sizes[node.id][0].borrow();
        generate_konst::<InDatum, OutDatum, _>(
            konst,
            output_shape,
            const_n_points,
            node.id,
            &*name,
            value_dest,
            type_dest,
        )?;
        impl_str.push_str(&*format!("let data_{} = self.{}.eval();\n", node.id, name));
    } else if let Some(_mat_mul) = node.op.as_any().downcast_ref::<MatMul>() {
        name = format!("mat_mul{}", node.id); //TODO: transpositions
        let (input1_size, input2_size, input_n_points, input1_id, input2_id) =
            if let Some(n_points) = const_n_points.get(&node.inputs[0].node) {
                (
                    input_sizes[0],
                    input_sizes[1],
                    n_points,
                    node.inputs[0].node,
                    node.inputs[1].node,
                )
            } else {
                if let Some(n_points) = const_n_points.get(&node.inputs[1].node) {
                    (
                        input_sizes[1],
                        input_sizes[0],
                        n_points,
                        node.inputs[1].node,
                        node.inputs[0].node,
                    )
                } else {
                    todo!()
                }
            };
        generate_matmul::<OutDatum, _>(
            input1_size,
            *input_n_points,
            input2_size,
            &*name,
            value_dest,
            type_dest,
        )?;
        impl_str.push_str(&*format!(
            "let data_{} = self.{}.eval(&data_{}, &data_{});\n",
            node.id, name, input1_id, input2_id
        ));
    } else if let Some(source) = node.op.as_any().downcast_ref::<TypedSource>() {
        input_str.push_str(&*format!("data_{}: ", node.id));
        for _ in 0..source.fact.shape.len() {
            input_str.push_str("[");
        }
        input_str.push_str(OutDatum::name());
        let mut v = source
            .fact
            .shape
            .eval(&SymbolValues::default())?
            .into_owned();
        v.reverse();
        for s in v.iter() {
            input_str.push_str(&*format!("; {}]", s));
        }
        input_str.push_str(",");
        return Ok(());
    } else if let Some(axis_op) = node.op.as_any().downcast_ref::<AxisOp>() {
        name = format!("axis_op{}", node.id);
        generate_axis_op::<OutDatum, _>(axis_op, &input_sizes[0], &*name, value_dest, type_dest)?;
        impl_str.push_str(&*format!(
            "let data_{} = self.{}.eval(data_{});\n",
            node.id, name, node.inputs[0].node
        ));
    } else {
        todo!()
    }
    writeln!(value_dest, ",")?;
    writeln!(type_dest, ",")?;
    Ok(())
}
fn generate_axis_op<OutDatum: Datum, W: Write>(
    axis_op: &AxisOp,
    in_size: &TVec<usize>,
    name: &str,
    value_dest: &mut W,
    type_dest: &mut W,
) -> TractResult<()> {
    if let AxisOp::Rm(axis) = axis_op {
        if in_size.len() == 4 && *axis == 3 {
            write!(
                type_dest,
                "{}: ::sparse_embedded::ops::axis_op::RmW<{}, {}, {}, {}>",
                name,
                OutDatum::name(),
                in_size[0],
                in_size[1],
                in_size[2]
            )?;
            write!(
                value_dest,
                "{}: ::sparse_embedded::ops::axis_op::RmW::DEFAULT",
                name,
            )?;
        } else if in_size.len() == 3 && *axis == 2 {
            write!(
                type_dest,
                "{}: ::sparse_embedded::ops::axis_op::RmH<{}, {}, {}>",
                name,
                OutDatum::name(),
                in_size[0],
                in_size[1],
            )?;
            write!(
                value_dest,
                "{}: ::sparse_embedded::ops::axis_op::RmH::DEFAULT",
                name,
            )?;
        } else {
            todo!()
        }
    } else {
        todo!()
    }
    Ok(())
}
fn generate_matmul<OutDatum: Datum, W: Write>(
    input1_shape: &TVec<usize>,
    input1_n_points: usize,
    input2_shape: &TVec<usize>,
    name: &str,
    value_dest: &mut W,
    type_dest: &mut W,
) -> TractResult<()> {
    write!(
        value_dest,
        "{}: ::sparse_embedded::ops::matmul::MatMul::DEFAULT",
        name
    )?;
    //currently only support multiplication between a sparse matrix and a full matrix.
    assert_eq!(input1_shape.len(), 2);
    assert_eq!(input2_shape.len(), 2);
    //TODO: proper error handling
    write!(
        type_dest,
        "{}: ::sparse_embedded::ops::matmul::MatMul<{}, {}, {}, {}, {}>",
        name,
        OutDatum::name(),
        input1_shape[0],
        input1_shape[1],
        input2_shape[0],
        input1_n_points
    )?;
    Ok(())
}
fn generate_konst<InDatum: Datum, OutDatum: Datum + From<InDatum>, W: Write>(
    tensor: &Arc<Tensor>,
    output_size: &TVec<usize>,
    const_n_points: &mut HashMap<usize, usize>,
    node_id: usize,
    name: &str,
    value_dest: &mut W,
    type_dest: &mut W,
) -> TractResult<()> {
    if output_size.len() == 2 {
        let mut n_points = 0;
        write!(
            value_dest,
            "{}: ::sparse_embedded::ops::konst::Const2 {{ data: [",
            name
        )?;
        let h_size = output_size[0];
        let w_size = output_size[1];
        let data = tensor.as_slice::<InDatum>()?;
        for h in 0..h_size {
            for w in 0..w_size {
                let value = &data[h * w_size + w];
                if *value == InDatum::default() {
                    continue;
                }
                let converted_value: OutDatum = value.clone().into();
                n_points += 1;
                write!(
                    value_dest,
                    "\n\t::sparse_embedded::ops::konst::DataPoint2 {{value: {}, h: {}, w:{}}},",
                    converted_value, h, w
                )?;
            }
        }
        const_n_points.insert(node_id, n_points);
        write!(value_dest, "]}}")?;
        write!(
            type_dest,
            "{}: ::sparse_embedded::ops::konst::Const2<{}, {}, {}, {}>",
            name,
            OutDatum::name(),
            h_size,
            w_size,
            n_points
        )?;
    } else if output_size.len() == 4 {
        let mut n_points = 0;
        write!(
            value_dest,
            "{}: ::sparse_embedded::ops::konst::Const4 {{ data: [",
            name
        )?;
        let o_size = output_size[0];
        let i_size = output_size[1];
        let h_size = output_size[2];
        let w_size = output_size[3];
        let data = tensor.as_slice::<InDatum>()?;
        for o in 0..o_size {
            for i in 0..i_size {
                for h in 0..h_size {
                    for w in 0..w_size {
                        let value = &data
                            [o * i_size * h_size * w_size + i * h_size * w_size + h * w_size + w];
                        if *value == InDatum::default() {
                            continue;
                        }
                        let converted_value: OutDatum = value.clone().into();
                        n_points += 1;
                        write!(
                            value_dest,
                            "\n\t::sparse_embedded::ops::konst::DataPoint4 {{value: {}, o: {}, i: {}, h: {}, w:{}}},",
                            converted_value, o, i, h, w
                        )?;
                    }
                }
            }
        }
        const_n_points.insert(node_id, n_points);
        write!(value_dest, "]}}")?;
        write!(
            type_dest,
            "{}: ::sparse_embedded::ops::konst::Const4<{}, {}, {}, {}, {}, {}>",
            name,
            OutDatum::name(),
            o_size,
            i_size,
            h_size,
            w_size,
            n_points
        )?;
    } else {
        todo!()
    }
    Ok(())
}
fn generate_max_pool<InDatum: Datum, OutDatum: Datum + From<InDatum>, W: Write>(
    input_size: &TVec<usize>,
    max_pool: &MaxPool,
    name: &str,
    value_dest: &mut W,
    type_dest: &mut W,
) -> TractResult<()> {
    let (input_shape, _patch, output_shape) = max_pool.pool_spec.compute_geo(input_size)?;
    let n = input_shape.n().unwrap_or(&1);
    let c = input_shape.c();
    let hi = input_shape.hw_dims()[0];
    let ho = output_shape.hw_dims()[0];
    let wi = input_shape.hw_dims()[1];
    let wo = output_shape.hw_dims()[1];
    let h_kernel = max_pool.pool_spec.kernel_shape[0];
    let w_kernel = max_pool.pool_spec.kernel_shape[1];
    let h_stride = max_pool.pool_spec.stride(0);
    let w_stride = max_pool.pool_spec.stride(1);
    write!(
        type_dest,
        "{}: ::sparse_embedded::ops::maxpool::MaxPool< {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
        name,
        OutDatum::name(),
        n,
        c,
        hi,
        ho,
        h_stride,
        h_kernel,
        wi,
        wo,
        w_stride,
        w_kernel
    )?;
    write!(
        value_dest,
        "{}: ::sparse_embedded::ops::maxpool::MaxPool::DEFAULT",
        name
    )?;
    Ok(())
}
fn generate_unary_op<OutDatum: Datum, W: Write>(
    un_op: &UnaryOp,
    in_size: &TVec<usize>,
    name: &str,
    value_dest: &mut W,
    type_dest: &mut W,
) -> TractResult<()> {
    if let Some(_) = un_op.mini_op.as_any().downcast_ref::<Max>() {
        //only relu is implemented, so it just checks if the constant is indeed 0.
        if un_op
            .a
            .close_enough(
                &Tensor::zero_dt(un_op.a.datum_type(), un_op.a.shape())?,
                true,
            )
            .is_ok()
        {
            if in_size.len() == 4 {
                write!(
                    value_dest,
                    "{}: ::sparse_embedded::ops::map::ReLu4::DEFAULT",
                    name
                )?;
                let (n, c, h, w) = (in_size[0], in_size[1], in_size[2], in_size[3]);
                write!(
                    type_dest,
                    "{}: ::sparse_embedded::ops::map::ReLu4<{}, {}, {}, {}, {}>",
                    name,
                    OutDatum::name(),
                    n,
                    c,
                    h,
                    w
                )?;
            } else if in_size.len() == 2 {
                write!(
                    value_dest,
                    "{}: ::sparse_embedded::ops::map::ReLu2::DEFAULT",
                    name
                )?;
                let (h, w) = (in_size[0], in_size[1]);
                write!(
                    type_dest,
                    "{}: ::sparse_embedded::ops::map::ReLu2<{}, {}, {}>",
                    name,
                    OutDatum::name(),
                    h,
                    w
                )?;
            } else {
                todo!()
            }
        } else {
            todo!()
        }
    } else {
        todo!()
    }
    Ok(())
}
fn write_kernel_point<DATUM: Datum>(
    dest: &mut impl Write,
    value: DATUM,
    out_feature: usize,
    in_feature: usize,
    kernel_x: usize,
    kernel_y: usize,
) -> TractResult<()> {
    if kernel_x >= 16 || kernel_y >= 16 {
        panic!("Kernel of size bigger than 16 currently unsupported")
    }
    if in_feature >= 256 {
        panic!("Maximum of 255 channels currently unsupported")
    }
    let i = kernel_x + 16 * kernel_y;
    write!(
        dest,
        "\n\t::sparse_embedded::ops::conv_unary::KernelPoint {{value: {:?}, i: {}, in_feature: {}, out_feature: {}}},",
        value, i, in_feature, out_feature
    )?;
    Ok(())
}

fn write_kernel<InDatum: Datum, OutDatum: Datum + From<InDatum>, W: Write>(
    dest: &mut W,
    kernel: &[InDatum],
    kernel_oihw: (usize, usize, usize, usize),
) -> TractResult<usize> {
    write!(dest, "kernel: [")?;
    let o_size = kernel_oihw.0;
    let i_size = kernel_oihw.1;
    let h_size = kernel_oihw.2;
    let w_size = kernel_oihw.3;
    let mut n_points = 0;
    for o in 0..o_size {
        for i in 0..i_size {
            for h in 0..h_size {
                for w in 0..w_size {
                    let value = kernel
                        [o * i_size * h_size * w_size + i * w_size * h_size + h * w_size + w]
                        .clone();
                    if value == InDatum::default() {
                        continue;
                    }
                    let converted_value: OutDatum = value.into();
                    n_points += 1;
                    write_kernel_point(dest, converted_value, o, i, h, w)?;
                }
            }
        }
    }
    write!(dest, "],\n")?;
    Ok(n_points)
}
fn generate_conv_unary<InDatum: Datum, OutDatum: Datum + From<InDatum>, W: Write>(
    conv: &ConvUnary,
    in_size: &TVec<usize>,
    name: &str,
    value_dest: &mut W,
    type_dest: &mut W,
) -> TractResult<()> {
    let kernel = conv.kernel_as_group_o_ihw()?;
    //TODO: support groups
    let shape = [kernel.shape()[0] * kernel.shape()[1], kernel.shape()[2]];
    let kernel = (*kernel).clone().into_shape(&shape)?;
    let (input_shape, _patch, output_shape) = conv.pool_spec.compute_geo(in_size)?;
    let n = input_shape.n().unwrap_or(&1);
    let ci = input_shape.c();
    let co = output_shape.c();
    let hi = input_shape.hw_dims()[0];
    let ho = output_shape.hw_dims()[0];
    let wi = input_shape.hw_dims()[1];
    let wo = output_shape.hw_dims()[1];
    let h_kernel = conv.pool_spec.kernel_shape[0];
    let w_kernel = conv.pool_spec.kernel_shape[1];
    write!(
        value_dest,
        "{}: ::sparse_embedded::ops::conv_unary::ConvUnary {{ ",
        name
    )?;
    let n_points = write_kernel::<InDatum, OutDatum, _>(
        value_dest,
        kernel.as_slice()?,
        (*co, *ci, h_kernel, w_kernel),
    )?;
    write!(value_dest, "}}")?;
    write!(
        type_dest,
        "{}: ::sparse_embedded::ops::conv_unary::ConvUnary<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
        name,
        OutDatum::name(),
        n,
        ci,
        co,
        hi,
        ho,
        h_kernel,
        wi,
        wo,
        w_kernel,
        n_points
    )?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use tract_core::prelude::Framework;
    use tract_onnx::{onnx, prelude::InferenceModelExt};

    use crate::generate;

    #[test]
    fn load() {
        let plan = onnx()
            .model_for_path("../net.onnx")
            .unwrap()
            .into_typed()
            .unwrap()
            .into_runnable()
            .unwrap();
        println!("---{:#?}", plan);
    }
    #[test]
    fn gen() {
        generate::<f32, f32, _>("../net.onnx", "net").unwrap();
    }
}
