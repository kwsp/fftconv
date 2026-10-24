benchmark_dep = dependency('benchmark')
armadillo_dep = dependency('armadillo')

executable('bench_fftconv',
  'bench_fftconv.cpp',
  'bench_hilbert.cpp',
  include_directories : fftconv_inc,
  dependencies : [fftw3_dep, fftw3f_dep, benchmark_dep, armadillo_dep],
)
