py = import('python').find_installation(pure: false)
pybind11_dep = dependency('pybind11')

py.extension_module(
  '_pyfftconv',
  'main.cpp',
  include_directories: fftconv_inc,
  dependencies: [fftw3_dep, fftw3f_dep, pybind11_dep],
  install: true,
  subdir: 'pyfftconv',
)

py.install_sources(
  'pyfftconv/__init__.py',
  'pyfftconv/__init__.pyi',
  'pyfftconv/_pyfftconv.pyi',
  subdir: 'pyfftconv',
)
