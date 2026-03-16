armadillo_dep = dependency('armadillo')
fmt_dep = dependency('fmt')
gtest_dep = dependency('gtest')

test_common_deps = [fftw3_dep, fftw3f_dep, fmt_dep, armadillo_dep]

executable('test_script_fftconv',
  'test_script.cpp',
  include_directories : fftconv_inc,
  dependencies : test_common_deps,
)

test_fftconv_exe = executable('test_fftconv',
  'test_fftw.cpp',
  'test_fftconv.cpp',
  'test_hilbert.cpp',
  include_directories : fftconv_inc,
  dependencies : test_common_deps + [gtest_dep],
)

test('fftconv', test_fftconv_exe, protocol : 'gtest')
