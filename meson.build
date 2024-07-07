project(
  'fftconv', 
  'cpp',
  'c',
  version : '0.1.0',
  default_options : [ # https://mesonbuild.com/Builtin-options.html
    'warning_level=3', 
    'cpp_std=c++14',
    'buildtype=release',
    'optimization=3',
    'debug=true',  # debug symbols
  ]
)
add_project_arguments('-Wno-sign-compare', language: 'cpp')


fftconv_src = [
  './fftconv_fftw/fftconv.cpp'
]

pocketfft_src = [
  'fftconv_pocketfft/pocketfft.c'
]
pocketfft_hdr = [
  'fftconv_pocketfft/pocketfft.h'
]

project_src = [
  fftconv_src,
  pocketfft_src
]

project_inc = include_directories(
  '.',
  './fftconv_fftw',
  './fftconv_pocketfft',
)

# Dependencies
cpp = meson.get_compiler('cpp')
larmadillo = cpp.find_library('armadillo', required: true)
lfftw3 = cpp.find_library('fftw3', required: true)
lbenchmark = cpp.find_library('benchmark', required: true)
project_dependencies = [
  dependency('threads'),
  larmadillo,
  lfftw3,
  lbenchmark,
]

# check for gperftools
lprofiler = cpp.find_library('profiler', required: true)


build_args = []

# ======
# Target
# ======

build_args += [
  '-DPROJECT_NAME=' + meson.project_name(),
  '-DPROJECT_VERSION=' + meson.project_version(),
]

executable(
  'test_fftconv',
  'test/test.cpp',
  project_src,
  dependencies: project_dependencies,
  include_directories: project_inc,
  install : false,
  c_args : build_args,
  cpp_args : build_args,
  link_args: '-Wl,--no-as-needed,-lprofiler,--as-needed' # link gperftools
)

executable(
  'bench_fftconv',
  'benchmark/bench_fftconv.cpp',
  project_src,
  dependencies: project_dependencies,
  include_directories: project_inc,
  install : false,
  c_args : build_args,
)

executable(
  'bench_pocketfft_hdr_thread',
  'benchmark/bench_pocketfft_hdr_thread.cpp',
  project_src,
  dependencies: project_dependencies,
  include_directories: project_inc,
  install : false,
  c_args : build_args,
)
