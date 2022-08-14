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
    iterations: int

    @classmethod
    def from_json(cls, bm) -> Benchmark:
        name = bm["name"]
        func_name, params = name.split("/", 1)
        params = params.split("/")
        assert bm["time_unit"] == "ns"
        return cls(
            name=name,
            func_name=func_name,
            param1=int(params[0]),
            param2=int(params[1]),
            cpu_time_ns=int(bm["cpu_time"]),
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
df = df[df.func_name.str.endswith("oaconvolve")]

# %%
for param2 in df["param2"].unique():
    fig, ax = plt.subplots()
    groupby = df[df["param2"] == param2].groupby("func_name")
    groupby.plot(x="param1", y="cpu_time_ns", ax=ax, marker="x", markeredgewidth=2.0)
    ax.legend(labels=groupby.groups.keys())
    ax.set_xlabel("Signal Length")
    ax.set_ylabel("Time (ns)")
    ax.set_title(f"Kernel length = {param2}")

# %%

# %%
