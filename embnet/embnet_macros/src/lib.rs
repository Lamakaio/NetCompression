use proc_macro::TokenStream;
use std::fs::read_to_string;
use std::io::Write;
use syn::{parse_macro_input, DeriveInput};
fn read_files(name: &str) -> std::io::Result<(String, String, String)> {
    let prefix = std::env::var("OUT_DIR").unwrap_or(String::new()) + "embnet_build/";
    let impl_str = read_to_string(prefix.clone() + &*format!("{}_impl.rs", name))?;
    let type_str = read_to_string(prefix.clone() + &*format!("{}_type.rs", name))?;
    let value_str = read_to_string(prefix.clone() + &*format!("{}_value.rs", name))?;
    Ok((impl_str, type_str, value_str))
}
//#[macro_export]
#[proc_macro_attribute]
pub fn net(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let name = input.ident.to_string();
    let net_name = attr.to_string();
    let (impl_str, type_str, value_str) = read_files(&*net_name).unwrap();
    let out = format!(
        "struct {} {{
    {}
}}
impl {} {{
    
    const NET:  Self = 
        {};
    {}
}}",
        name, type_str, name, value_str, impl_str
    )
    .parse()
    .unwrap();
    write!(std::fs::File::create("out").unwrap(), "{}", out).unwrap();
    out
}
