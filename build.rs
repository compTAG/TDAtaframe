// build.rs
use std::env;
use std::path::Path;
use std::process::Command;
use std::str;

fn get_venv_torch_libdir() -> String {
    // Use the Python executable from the current environment (venv)
    let python_exe = env::var("PYTHON_SYS_EXECUTABLE")
        .unwrap_or_else(|_| "python".to_string());

    let py_snippet = r#"
import sys
import pathlib
try:
    import torch
except ImportError:
    sys.stderr.write("ERROR: Python package 'torch' not found. Please install it in the current venv.\n")
    sys.exit(1)

lib_dir = pathlib.Path(torch.__file__).resolve().parent / "lib"
print(str(lib_dir))
"#;

    let output = Command::new(&python_exe)
        .arg("-c")
        .arg(py_snippet)
        .output()
        .expect("Failed to execute Python to locate torch libdir");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("Python execution failed:\n{}", stderr);
    }

    let libdir = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !Path::new(&libdir).exists() {
        panic!("Torch lib dir '{}' does not exist", libdir);
    }

    libdir
}

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| "unknown".into());

    if target_os != "linux" {
        panic!("This build.rs is Linux-only for now");
    }

    let torch_libdir = get_venv_torch_libdir();
    println!("cargo:warning=Using torch lib dir from Python venv: {}", torch_libdir);

    // Add to linker search path
    println!("cargo:rustc-link-search=native={}", torch_libdir);

    // Add rpath so loader finds libtorch at runtime
    println!("cargo:rustc-link-arg=-Wl,-rpath={}", torch_libdir);

    // Standard linker flags for Linux
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-Wl,--copy-dt-needed-entries");

    // Link against libtorch
    println!("cargo:rustc-link-arg=-ltorch");
}

