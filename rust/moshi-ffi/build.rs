use std::env;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let mut config = cbindgen::Config::default();
    config.language = cbindgen::Language::C;
    config.enumeration.prefix_with_name = true;

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_include_guard("MOSHI_FFI_H")
        .with_documentation(true)
        .with_config(config)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("include/moshi.h");
}
