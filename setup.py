import os
import sys
import glob
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
candidate_ducc_roots = [
    os.environ.get("DUCCDIR", ""),
    "/fred/oz451/azic/scripts/src/ducc",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ducc")),
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ducc")),
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src", "ducc")),
]

ducc_found = False
for cand in candidate_ducc_roots:
    if cand and os.path.isdir(cand):
        ducc_src = os.path.join(cand, "src")
        for lib_sub in ["build_cpp", "build", "build_cpp/src", "build/src", ""]:
            cand_lib = os.path.join(cand, lib_sub) if lib_sub else cand
            target_a = os.path.join(cand_lib, "libducc0.a")
            target_so = os.path.join(cand_lib, "libducc0.so")

            # Remove any symlink libducc0.so -> ducc0.cpython...so because python module shared
            # objects hide C++ internal symbols (-fvisibility=hidden). Static archive is required.
            if os.path.islink(target_so):
                try:
                    os.unlink(target_so)
                except Exception:
                    pass

            # If libducc0.a does not exist, automatically archive it from compiled object files
            if not os.path.isfile(target_a):
                cmake_dir = os.path.join(cand_lib, "CMakeFiles", "ducc0.dir")
                if os.path.isdir(cmake_dir):
                    import subprocess
                    obj_files = []
                    for root, _, files in os.walk(cmake_dir):
                        for f in files:
                            if f.endswith(".o"):
                                obj_files.append(os.path.join(root, f))
                    if obj_files:
                        print(f"[setup.py] Creating static archive {target_a} from {len(obj_files)} object files...")
                        subprocess.check_call(["ar", "rcs", target_a] + obj_files)

            if os.path.isfile(target_a) or os.path.isfile(target_so):
                include_dirs.append(ducc_src if os.path.isdir(ducc_src) else cand)
                library_dirs.append(cand_lib)
                extra_link_args.append(f"-Wl,-rpath,{cand_lib}")
                ducc_found = True
                print(f"[setup.py] Found ducc headers at {ducc_src}, library at {cand_lib}")
                break
        if not ducc_found and os.path.isdir(cand):
            # Fallback if library hasn't been built yet or has custom name
            ducc_lib = os.path.join(cand, "build_cpp")
            if os.path.isdir(ducc_src):
                include_dirs.append(ducc_src)
            if os.path.isdir(ducc_lib):
                library_dirs.append(ducc_lib)
                extra_link_args.append(f"-Wl,-rpath,{ducc_lib}")
                ducc_found = True
                print(f"[setup.py] Using fallback ducc at {cand}, lib at {ducc_lib}")
        if ducc_found:
            break

if not ducc_found:
    print("[setup.py] WARNING: Could not find DUCC directory or libducc0 archive!")

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
