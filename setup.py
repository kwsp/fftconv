from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "fftconv",
        sources=["./pyfftconv/*.pyx", "./src/fftconv.cpp"],
        include_dirs=["./src", "/opt/homebrew/include"],
        libraries=["fftw3"],
        library_dirs=["/opt/homebrew/lib"],
        extra_compile_args=["-std=c++17", "-O3", "-Wno-sign-compare"],
        language="c++",
    ),
]

setup(
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
