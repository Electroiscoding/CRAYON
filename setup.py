from setuptools import setup, Extension, find_packages

module = Extension(
    "crayon.crayon_module",  # Renamed to place directly under crayon package
    sources=[
        "src/crayon/c_ext/crayon_module.c",
        "src/crayon/c_ext/simd_ops.c",
    ],
    include_dirs=["src/crayon/c_ext"],
)

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[module],
)
