from setuptools import setup, Extension
from Cython.Build import cythonize

ext_libraries = [
    ["pocketfft", {
        "sources": ["./fftconv_pocketfft/pocketfft.c"],
        "include_dirs": ["./fftconv_pocketfft"],
    }]
]

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
            "./fftconv_pocketfft",
            "/opt/homebrew/include",
        ],
        libraries=["fftw3", "pocketfft"],
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
    libraries=ext_libraries,
)
