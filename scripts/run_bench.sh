#meson build && ninja -C build
#echo

#BIN=bench_fftconv
#./build/$BIN --benchmark_out=./$BIN.json --benchmark_out_format=json

#echo
#echo Generating plot
#python3 plot_bench.py

cmake --preset clang
cmake --build --preset clang-release --target bench_fftconv

BIN=bench_fftconv
DIR=./build/clang/benchmark/Release
$DIR/$BIN --benchmark_out=./$BIN.json --benchmark_out_format=json
