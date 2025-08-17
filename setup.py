import os
import sys
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
extra_link_args = []
libraries = ["fftw3f"]   # we only need the float API

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

common_includes = ["/usr/include", "/usr/local/include"]
common_libs = ["/usr/lib", "/usr/local/lib", "/usr/lib64", "/usr/local/lib64"]

for d in common_includes:
    if os.path.isdir(d):
        include_dirs.append(d)

for d in common_libs:
    if os.path.isdir(d):
        library_dirs.append(d)

# De-duplicate
include_dirs = list(dict.fromkeys(include_dirs))
library_dirs = list(dict.fromkeys(library_dirs))

ext_modules = [
    Pybind11Extension(
        "riptide.libcpp",
        sorted(["src/riptide/cpp/python_bindings.cpp"]),
        extra_compile_args=["-O3", "-march=native"] + SAFE_FAST_MATH_FLAGS,
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
