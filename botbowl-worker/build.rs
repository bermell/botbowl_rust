//! Plan 066: with the `torch` feature, `botbowl-nn` links libtorch and publishes where it lives
//! (`DEP_BBNN_TORCH_RPATH`); give this crate's binaries an rpath to it, so they run without
//! `LD_LIBRARY_PATH`. Without the feature the variable is unset and this does nothing.
fn main() {
    println!("cargo:rerun-if-env-changed=DEP_BBNN_TORCH_RPATH");
    if let Some(dirs) = std::env::var_os("DEP_BBNN_TORCH_RPATH") {
        for d in std::env::split_paths(&dirs) {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", d.display());
        }
    }
}
