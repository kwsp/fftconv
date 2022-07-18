from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "fftconv",
        sources=["./pyfftconv/*.pyx"],
        include_dirs=["include", "/opt/homebrew/include"],
        libraries=["fftw3"],
        library_dirs=["/opt/homebrew/lib"],
        extra_compile_args=["-std=c++17", "-O3"],
        language="c++",
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
        annotate=True,
    ),
)
