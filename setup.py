import os
import sys
import glob
import shutil
import platform
import subprocess
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

SAFE_FAST_MATH_FLAGS = [
    "-fassociative-math",
    "-fno-math-errno",
    "-ffinite-math-only",
    "-fno-rounding-math",
    "-fno-signed-zeros",
    "-fno-trapping-math",
]

include_dirs = []
library_dirs = []
extra_link_args = ["-pthread", "-fno-lto"]
extra_compile_args = ["-O3", "-std=c++17", "-pthread", "-fno-lto"] + SAFE_FAST_MATH_FLAGS

if platform.machine() in ("x86_64", "AMD64"):
    # Target AVX2 / FMA baseline supported by both Intel Skylake and AMD Milan;
    # explicitly disable AVX-512 to avoid SIGILL when built on Skylake login nodes.
    extra_compile_args.extend(["-march=x86-64-v3", "-mno-avx512f"])

libraries = []
cpp_sources = ["src/riptide/cpp/python_bindings.cpp"]

# Path to local ducc static library build and headers
candidate_ducc_roots = [
    os.environ.get("DUCCDIR", ""),
    "/fred/oz451/azic/scripts/src/ducc",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ducc")),
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ducc")),
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src", "ducc")),
]

# Prefer compiling DUCC C++ sources directly into libcpp.
# This eliminates symbol visibility issues, LTO bytecode conflicts, and external library linking errors.
ducc_sources = []
for cand in candidate_ducc_roots:
    if cand and os.path.isdir(cand):
        ducc_src = os.path.join(cand, "src")
        ducc0_dir = os.path.join(ducc_src, "ducc0")
        if os.path.isdir(ducc0_dir):
            include_dirs.append(ducc_src)
            for root, _, files in os.walk(ducc0_dir):
                for f in files:
                    if f.endswith(".cc"):
                        ducc_sources.append(os.path.join(root, f))
            if ducc_sources:
                print(f"[setup.py] Compiling {len(ducc_sources)} DUCC C++ source files directly into libcpp from {ducc_src}")
                cpp_sources.extend(sorted(ducc_sources))
                break

if not ducc_sources:
    # Fallback to external library if sources cannot be found
    libraries.append("ducc0")
    for cand in candidate_ducc_roots:
        if cand and os.path.isdir(cand):
            ducc_src = os.path.join(cand, "src")
            ducc_lib = os.path.join(cand, "build_cpp")
            if os.path.isdir(ducc_src):
                include_dirs.append(ducc_src)
            if os.path.isdir(ducc_lib):
                library_dirs.append(ducc_lib)
                extra_link_args.append(f"-Wl,-rpath,{ducc_lib}")
            break

fftwd_dir = os.environ.get("FFTWDIR")
if fftwd_dir:
    include_dirs.append(os.path.join(fftwd_dir, "include"))
    library_dirs.append(os.path.join(fftwd_dir, "lib"))
    extra_link_args.append(f"-Wl,-rpath,{os.path.join(fftwd_dir, 'lib')}")

conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    include_dirs.append(os.path.join(conda_prefix, "include"))
    library_dirs.append(os.path.join(conda_prefix, "lib"))
    extra_link_args.append(f"-Wl,-rpath,{os.path.join(conda_prefix, 'lib')}")

common_includes = [
    os.path.join(sys.prefix, "include"),
    "/opt/homebrew/include",
    "/usr/include",
    "/usr/local/include",
]
common_libs = [
    os.path.join(sys.prefix, "lib"),
    "/opt/homebrew/lib",
    "/usr/lib",
    "/usr/local/lib",
    "/usr/lib64",
    "/usr/local/lib64",
]

for d in common_includes:
    if os.path.isdir(d):
        include_dirs.append(d)

for d in common_libs:
    if os.path.isdir(d):
        library_dirs.append(d)
        extra_link_args.append(f"-Wl,-rpath,{d}")

# De-duplicate
include_dirs = list(dict.fromkeys(include_dirs))
library_dirs = list(dict.fromkeys(library_dirs))

# Conditional CUDA compilation when nvcc is available and not opted out
disable_cuda = os.environ.get("RIPTIDE_NO_CUDA", "").strip().lower() in ("1", "true", "yes", "on")
nvcc_path = None if disable_cuda else shutil.which("nvcc")
if not disable_cuda and nvcc_path is None:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        cand_nvcc = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.isfile(cand_nvcc) and os.access(cand_nvcc, os.X_OK):
            nvcc_path = cand_nvcc

cuda_source = os.path.join("src", "riptide", "cpp", "cuda_transforms.cu")
use_cuda = (nvcc_path is not None) and os.path.isfile(cuda_source)
define_macros = []

if use_cuda:
    define_macros.append(("HAVE_CUDA", "1"))
    cuda_roots = []
    for env_key in ("CUDA_HOME", "CUDA_PATH"):
        val = os.environ.get(env_key)
        if val:
            cuda_roots.append(val)
    cuda_roots.append(os.path.dirname(os.path.dirname(nvcc_path)))
    cuda_roots.append(os.path.dirname(os.path.dirname(os.path.realpath(nvcc_path))))
    for root in cuda_roots:
        for sub in ("lib64", "lib"):
            cand_lib = os.path.join(root, sub)
            if os.path.isdir(cand_lib):
                library_dirs.append(cand_lib)
                extra_link_args.append(f"-Wl,-rpath,{cand_lib}")
    library_dirs = list(dict.fromkeys(library_dirs))
    extra_link_args = list(dict.fromkeys(extra_link_args))
    libraries.append("cudart")


class CudaBuildExt(build_ext):
    """Build extension subclass compiling ``cuda_transforms.cu`` with ``nvcc`` when enabled."""

    def build_extensions(self):
        if use_cuda:
            os.makedirs(self.build_temp, exist_ok=True)
            cuda_obj = os.path.join(self.build_temp, "cuda_transforms.o")
            nvcc_cmd = [
                nvcc_path,
                "-c",
                cuda_source,
                "-o",
                cuda_obj,
                "-std=c++17",
                "-O3",
                "-Xcompiler",
                "-fPIC",
                "-gencode",
                "arch=compute_70,code=sm_70",
                "-gencode",
                "arch=compute_80,code=sm_80",
                "-gencode",
                "arch=compute_80,code=compute_80",
            ]
            print(f"[setup.py] Compiling CUDA kernel object: {' '.join(nvcc_cmd)}")
            subprocess.check_call(nvcc_cmd)
            for ext in self.extensions:
                if ext.name == "riptide.libcpp":
                    objs = list(ext.extra_objects or [])
                    if cuda_obj not in objs:
                        objs.append(cuda_obj)
                    ext.extra_objects = objs
        super().build_extensions()


ext_modules = [
    Pybind11Extension(
        "riptide.libcpp",
        sorted(cpp_sources),
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]

if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
        cmdclass={"build_ext": CudaBuildExt},
    )
