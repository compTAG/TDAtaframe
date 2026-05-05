fn main() {
    let os = std::env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");

    match os.as_str() {
        "linux" => {
            if let Some(lib_path) = std::env::var_os("DEP_TCH_LIBTORCH_LIB") {
                println!(
                    "cargo:rustc-link-arg=-Wl,-rpath={}",
                    lib_path.to_string_lossy()
                );
            }
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            println!("cargo:rustc-link-arg=-ltorch");
        }
        "macos" => {
            if let Some(lib_path) = std::env::var_os("DEP_TCH_LIBTORCH_LIB") {
                println!(
                    "cargo:rustc-link-arg=-Wl,-rpath,{}",
                    lib_path.to_string_lossy()
                );
            }
        }
        "windows" => {
            if let Some(lib_path) = std::env::var_os("DEP_TCH_LIBTORCH_LIB") {
                // On Windows the rpath concept doesn't apply; the DLLs need to
                // be on PATH at runtime.  We just print it so maturin can copy
                // them into the wheel via the delvewheel/repair step.
                println!("cargo:warning=libtorch lib path: {}", lib_path.to_string_lossy());
            }
        }
        _ => {}
    }
}
