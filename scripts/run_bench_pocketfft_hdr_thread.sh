meson build && ninja -C build
echo

BIN=bench_pocketfft_hdr_thread
./build/$BIN --benchmark_out=./$BIN.json --benchmark_out_format=json

echo
echo Generating plot
python3 plot_bench.py
