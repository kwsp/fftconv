#!/bin/bash
meson build && ninja -C build
echo

./build/fftconv_bench --benchmark_out=./bench_result.json --benchmark_out_format=json
