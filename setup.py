from __future__ import annotations

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


ext_modules = [
    Pybind11Extension(
        "_spatial_native",
        ["native/cpp_probe_module.cpp"],
        cxx_std=17,
        extra_compile_args=["/O2"] if __import__("sys").platform == "win32" else ["-O3", "-g0"],
    )
]


setup(
    py_modules=["cpp_probe"],
    packages=[],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
