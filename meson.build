project('fftconv', 'c', 'cpp',
  version : '0.5.1',
  default_options : ['cpp_std=c++20', 'warning_level=2'],
)

fftconv_inc = include_directories('include')

fftw3_dep = dependency('fftw3')
fftw3f_dep = dependency('fftw3f')

subdir('test')
subdir('benchmark')
subdir('py')
