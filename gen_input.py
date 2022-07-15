import numpy as np


def save_test_case(case_id, v1, v2, gt):
    s = ",".join((f"{x:.2f}" for x in v1))
    s += "\n"
    s += ",".join((f"{x:.2f}" for x in v2))
    s += "\n"
    s += ",".join((f"{x:.2f}" for x in gt))
    with open(f"test_case_{case_id}.txt", "w") as fp:
        fp.write(s)

# case 1: 1664 ** 65
v1 = 10 * np.random.random(1664)
v2 = 10 * np.random.random(65)
save_test_case(1, v1, v2, [123])

# case 2: 2816 ** 65
v1 = 10 * np.random.random(2816)
v2 = 10 * np.random.random(65)
save_test_case(2, v1, v2, [123])

# case 3: 10 ** 5
v1 = 10 * np.random.random(10)
v2 = 10 * np.random.random(5)
save_test_case(3, v1, v2, [123])
