import os
import sys
import platform
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
extra_link_args = ["-pthread"]
extra_compile_args = ["-O3", "-std=c++17", "-pthread"] + SAFE_FAST_MATH_FLAGS

if platform.machine() in ("x86_64", "AMD64"):
    # Target AVX2 / FMA baseline supported by both Intel Skylake and AMD Milan;
    # explicitly disable AVX-512 to avoid SIGILL when built on Skylake login nodes.
    extra_compile_args.extend(["-march=x86-64-v3", "-mno-avx512f"])

libraries = ["ducc0"]

# Path to local ducc static library build and headers
ducc_root = os.environ.get(
    "DUCCDIR",
    os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "ducc",
        )
    ),
)
if os.path.isdir(ducc_root):
    ducc_src = os.path.join(ducc_root, "src")
    ducc_lib = os.path.join(ducc_root, "build_cpp")
    if os.path.isdir(ducc_src):
        include_dirs.append(ducc_src)
    if os.path.isdir(ducc_lib):
        library_dirs.append(ducc_lib)
        extra_link_args.append(f"-Wl,-rpath,{ducc_lib}")

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

ext_modules = [
    Pybind11Extension(
        "riptide.libcpp",
        sorted(["src/riptide/cpp/python_bindings.cpp"]),
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
        cmdclass={"build_ext": build_ext},
    )
