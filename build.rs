use std::env;
use std::path::Path;
use std::process::Command;

fn get_venv_torch_libdir() -> String {
    let python_exe = env::var("PYTHON_SYS_EXECUTABLE").unwrap_or_else(|_| "python".to_string());

    let py_snippet = r#"
import sys, pathlib
try:
    import torch
except ImportError:
    sys.stderr.write("ERROR: Python package 'torch' not found.\n")
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
    let torch_libdir = get_venv_torch_libdir();
    println!("cargo:warning=Using torch lib dir from Python venv: {}", torch_libdir);

    // Add search path for all platforms
    println!("cargo:rustc-link-search=native={}", torch_libdir);

    match target_os.as_str() {
        "linux" => {
            println!("cargo:rustc-link-arg=-Wl,-rpath={}", torch_libdir);
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            println!("cargo:rustc-link-arg=-Wl,--copy-dt-needed-entries");
            println!("cargo:rustc-link-arg=-ltorch");
        }
        "macos" => {
            // On macOS, we need to specify the torch lib directory for linking and runtime
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", torch_libdir);
            println!("cargo:rustc-link-arg=-ltorch"); // links libtorch.dylib
        }
        "windows" => {
            // On Windows, we need to ensure the torch library path is properly set
            // The torch library directory is already added to the search path above
            println!("cargo:rustc-link-lib=dylib=torch"); // links torch.dll
        }
        _ => panic!("Unsupported OS: {}", target_os),
    }
}
