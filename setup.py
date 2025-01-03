from setuptools import setup, Extension
from Cython.Build import cythonize


extensions = [
    Extension(
        "fftconv",
        sources=[
            "./pyfftconv/*.pyx",
            "./fftconv_fftw/fftconv.cpp",
        ],
        include_dirs=[
            "./",
            "./fftconv_fftw",
            "/opt/homebrew/include",
        ],
        libraries=["fftw3"],
        library_dirs=["/opt/homebrew/lib"],
        extra_compile_args=["-std=c++17", "-Ofast", "-Wno-sign-compare"],
        language="c++",
    ),
]

setup(
    name="fftconv",
    install_requires=[
        "numpy",
        "scipy",
    ],
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
        annotate=True,
    ),
)
