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
    cpu_time_ns: int
    cpu_time_ms: int
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
            cpu_time_ns=time_ns,
            cpu_time_ms=int(time_ns/1000),
            iterations=int(bm["iterations"]),
        )

#%%
with open("./bench_result.json") as fp:
    data = json.load(fp)

context = data["context"]
date = context["date"].rsplit("-", 1)[0]
benchmark = data["benchmarks"][0]
# pprint(benchmark)

bms = [Benchmark.from_json(bm) for bm in data["benchmarks"]]
df = pd.DataFrame(bms)
df.head()

# %%
df = df[df.func_name.str.startswith("BM_oaconvolve")]
# df = df[df.func_name.str.endswith("naive") == False]
df

# %%
param2unique = df["param2"].unique()
assert len(param2unique) == 4 # change subplot otherwise
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
for ax, param2 in zip(axes.flatten(), param2unique):
    # group by param2 (second input to bench range, kernel length)
    groupby = df[df["param2"] == param2].groupby("func_name")
    # plot CPU time
    groupby.plot(x="param1", y="cpu_time_ms", ax=ax, marker="x", markeredgewidth=2.0)
    ax.set_xlabel("Signal Length")
    ax.set_ylabel("Time (ms)")
    ax.set_ylim(ymin=0)
    # Remove BM_prefix from function name for label
    labels = [k.removeprefix("BM_") for k in groupby.groups.keys()]
    ax.legend(labels)
    ax.set_title(f"Kernel length = {param2}")

fig.suptitle("Overlap-Add Convolution (C++)")
plt.savefig(f"bench_{date}.svg")

# %%
