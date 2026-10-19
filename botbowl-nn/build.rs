//! Builds the libtorch shim (`csrc/inproc_shim.cpp`) when the `torch` feature is on (plan 066).
//! Without the feature this does nothing, so a default build needs no libtorch, no CUDA and no
//! C++ compiler beyond what the rest of the workspace needs.
//!
//! Where libtorch comes from, in order:
//! - `BOTBOWL_TORCH_DIR`: a directory with `include/` and `lib/` — the `torch` package of a Python
//!   wheel (`…/site-packages/torch`) or an unpacked libtorch;
//! - `<checkout>/train/.venv/lib/python3*/site-packages/torch` — the trainer's own wheel, which is
//!   what the sidecar runs, so the two share every kernel (a worktree finds its main checkout's).
//!
//! CUDA runtime headers (`cuda_runtime_api.h`): `BOTBOWL_CUDA_INCLUDE`, else the wheel's sibling
//! `nvidia/cuda_runtime/include`, else `/usr/local/cuda/include`. The binary gets an rpath to the
//! torch `lib/` (and to the CUDA runtime it linked), so it runs without `LD_LIBRARY_PATH`.
//! `BOTBOWL_TORCH_CXX11_ABI` (default 1; the 2.x wheels use the new ABI) sets
//! `_GLIBCXX_USE_CXX11_ABI`.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "torch")]
    torch::build();
}

#[cfg(feature = "torch")]
mod torch {
    use std::path::{Path, PathBuf};

    fn find_torch() -> PathBuf {
        println!("cargo:rerun-if-env-changed=BOTBOWL_TORCH_DIR");
        if let Ok(d) = std::env::var("BOTBOWL_TORCH_DIR") {
            return PathBuf::from(d);
        }
        let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
        // A worktree under `.claude/worktrees/<name>/` has no venv of its own: look up the tree.
        for root in manifest.ancestors().skip(1) {
            let lib = root.join("train/.venv/lib");
            let Ok(rd) = std::fs::read_dir(&lib) else { continue };
            let mut pys: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
            pys.sort();
            for py in pys.into_iter().rev() {
                let t = py.join("site-packages/torch");
                if t.join("include").is_dir() && t.join("lib").is_dir() {
                    return t;
                }
            }
        }
        panic!(
            "feature `torch`: no libtorch found — set BOTBOWL_TORCH_DIR to a torch package dir \
             (with include/ and lib/), or create train/.venv (cd train && uv sync)"
        );
    }

    /// The include dirs for `cuda_runtime.h`: the first holds `cuda_runtime_api.h` (and its
    /// sibling `lib/` the runtime library), the rest supply `crt/host_config.h` when the first
    /// does not (the wheel's runtime package leaves crt/ to the nvcc package).
    fn find_cuda_include(torch: &Path) -> Vec<PathBuf> {
        println!("cargo:rerun-if-env-changed=BOTBOWL_CUDA_INCLUDE");
        if let Ok(d) = std::env::var("BOTBOWL_CUDA_INCLUDE") {
            return std::env::split_paths(&d).collect();
        }
        let wheel = torch.parent().map(|p| p.join("nvidia/cuda_runtime/include"));
        let Some(rt) = [wheel, Some(PathBuf::from("/usr/local/cuda/include"))]
            .into_iter()
            .flatten()
            .find(|p| p.join("cuda_runtime_api.h").is_file())
        else {
            return Vec::new();
        };
        let mut dirs = vec![rt.clone()];
        // The wheel splits the toolkit's headers by library (cusparse.h, cublas_v2.h, cudnn.h …,
        // all reached from ATen's CUDA headers): every `nvidia/<lib>/include` beside the runtime.
        if let Some(nvidia) = rt.parent().and_then(|p| p.parent()) {
            if let Ok(rd) = std::fs::read_dir(nvidia) {
                let mut v: Vec<PathBuf> = rd
                    .filter_map(|e| e.ok().map(|e| e.path().join("include")))
                    .filter(|p| p.is_dir() && *p != rt)
                    .collect();
                v.sort();
                dirs.extend(v);
            }
        }
        if !rt.join("crt/host_config.h").is_file() {
            // `scripts/fetch_cuda_headers.sh` puts the nvcc wheel's include/ here; a venv that
            // has nvidia-cuda-nvcc-cu12 installed has it beside the runtime.
            let mut cands: Vec<PathBuf> = torch
                .parent()
                .map(|p| p.join("nvidia/cuda_nvcc/include"))
                .into_iter()
                .collect();
            if let Some(home) = std::env::var_os("HOME") {
                let cache = PathBuf::from(home).join(".cache/botbowl/cuda_nvcc");
                if let Ok(rd) = std::fs::read_dir(&cache) {
                    let mut v: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path().join("include"))).collect();
                    v.sort();
                    cands.extend(v.into_iter().rev());
                }
            }
            match cands
                .into_iter()
                .find(|p| p.join("crt/host_config.h").is_file() && p.join("nv/target").is_file())
            {
                Some(p) => dirs.push(p),
                None => panic!(
                    "feature `torch`: {} has no crt/host_config.h (nvcc) and nv/target (cccl) — run scripts/fetch_cuda_headers.sh \
                     (fetches the nvcc and cccl wheels' headers into ~/.cache/botbowl/cuda_nvcc), or set \
                     BOTBOWL_CUDA_INCLUDE to a CUDA toolkit's include/",
                    rt.display()
                ),
            }
        }
        dirs
    }

    pub fn build() {
        println!("cargo:rerun-if-changed=csrc/inproc_shim.cpp");
        println!("cargo:rerun-if-env-changed=BOTBOWL_TORCH_CXX11_ABI");
        let torch = find_torch();
        let inc = torch.join("include");
        let lib = torch.join("lib");
        let cuda_incs = find_cuda_include(&torch);
        let cuda_inc = cuda_incs.first().cloned().expect(
            "feature `torch`: cuda_runtime_api.h not found — set BOTBOWL_CUDA_INCLUDE (the wheel's \
             nvidia/cuda_runtime/include, or a CUDA toolkit's include/)",
        );
        let abi = std::env::var("BOTBOWL_TORCH_CXX11_ABI").unwrap_or_else(|_| "1".into());
        let mut build = cc::Build::new();
        build
            .cpp(true)
            .file("csrc/inproc_shim.cpp")
            .include(&inc)
            .include(inc.join("torch/csrc/api/include"));
        for d in &cuda_incs {
            build.include(d);
        }
        build
            .flag("-std=c++17")
            .flag("-w")
            .define("_GLIBCXX_USE_CXX11_ABI", abi.as_str())
            .opt_level(2)
            .compile("bbnn_inproc_shim");

        println!("cargo:rustc-link-search=native={}", lib.display());
        for l in ["torch", "torch_cpu", "torch_cuda", "c10", "c10_cuda"] {
            println!("cargo:rustc-link-lib=dylib={l}");
        }
        // The shim's inline CUDA event calls need the runtime itself. The wheel ships only the
        // versioned soname, so link it by file name.
        let cudart_dir = cuda_inc.parent().map(|p| p.join("lib")).filter(|p| p.is_dir());
        let cudart = cudart_dir.as_ref().and_then(|d| {
            std::fs::read_dir(d)
                .ok()?
                .filter_map(|e| e.ok())
                .map(|e| e.file_name())
                .find(|n| n.to_str().is_some_and(|s| s.starts_with("libcudart.so")))
        });
        let mut rpath = vec![lib.clone()];
        match (cudart_dir, cudart) {
            (Some(dir), Some(name)) => {
                println!("cargo:rustc-link-search=native={}", dir.display());
                println!("cargo:rustc-link-lib=dylib:+verbatim={}", name.to_string_lossy());
                rpath.push(dir);
            }
            _ => println!("cargo:rustc-link-lib=dylib=cudart"),
        }
        println!("cargo:rustc-link-lib=dylib=stdc++");
        // `rustc-link-arg` reaches only this package's own targets, so the rpath goes two ways:
        // directly (this crate's tests and examples), and as `links` metadata
        // (`DEP_BBNN_TORCH_RPATH`) for the binaries of crates that depend on this one — their
        // build scripts (`botbowl-ui/build.rs`, `botbowl-worker/build.rs`) add it to their link.
        for d in &rpath {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", d.display());
        }
        let joined = std::env::join_paths(&rpath).expect("rpath dirs");
        println!("cargo:rpath={}", joined.to_string_lossy());
    }
}
