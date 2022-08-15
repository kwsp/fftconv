#%%
from __future__ import annotations
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint

@dataclass
class Benchmark:
    name: str
    func_name: str
    param1: int
    param2: int
    param3: int
    cpu_time_ns: int
    cpu_time_ms: int
    real_time_ms: int
    iterations: int

    @classmethod
    def from_json(cls, bm) -> Benchmark:
        name = bm["name"]
        func_name, params = name.split("/", 1)
        params = params.split("/")
        assert bm["time_unit"] == "ns"
        time_ns = int(bm["cpu_time"])
        return cls(
            name=name,
            func_name=func_name,
            param1=int(params[0]),
            param2=int(params[1]),
            param3=int(params[2]),
            cpu_time_ns=time_ns,
            cpu_time_ms=int(time_ns/1000),
            real_time_ms=int(bm["real_time"]/1000),
            iterations=int(bm["iterations"]),
        )

#%%
fname = "./bench_pocketfft_hdr_thread.json"
with open(fname) as fp:
    data = json.load(fp)

context = data["context"]
date = context["date"].rsplit("-", 1)[0]
benchmark = data["benchmarks"][0]
pprint(benchmark)

bms = [Benchmark.from_json(bm) for bm in data["benchmarks"]]
df = pd.DataFrame(bms)
df.head()

# %%

# %%
param2s = df["param2"].unique()
assert len(param2s) == 1
param2 = param2s[0]

param1s = df["param1"].unique()
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
for ax, param1 in zip(axes.flatten(), param1s):
    df[df["param1"] == param1].plot.bar(x="param3", y="real_time_ms", ax=ax)
    ax.set_ylabel("Real time (ms)")
    ax.set_xlabel("n threads")
    ax.set_title(f"convolve_pocketfft ({param1}, {param2})")
    ax.legend([])
fig.suptitle("convolve_pocketfft_hdr scaling with nthreads")
plt.savefig(f"{fname}.svg")


# %%
