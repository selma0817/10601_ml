'''
Run this file to check your code's outputs against the 
ones given in reference_output/

Usage to run all checks: 
    python3 -m unittest check 
Usage to run check for specific example: 
    python3 -m unittest check.MCRaw
The examples (in the order they are given in the handout) are MCRaw, GWSimple, 
GW, and MCTile.

This file should be located in and run from the same directory as both
q_learning.py and the reference_output folder.
'''

import os
import subprocess
import sys
import time
import unittest
from numpy.testing import assert_allclose
import numpy as np

BASE_PY = sys.executable
BASE_FILE = "q_learning.py"

def run_command(env, mode, weight_out, returns_out, episodes, max_iterations, 
                epsilon, gamma, lr):
    """
    Runs a single command with the desired outputs
    """

    episodes, max_iterations, epsilon, gamma, lr = \
        (str(v) for v in (episodes, max_iterations, epsilon, gamma, lr))

    return subprocess.run([BASE_PY, BASE_FILE, env, mode, weight_out, returns_out, 
                           episodes, max_iterations, epsilon, gamma, lr], capture_output=True)

configs = [
    ("mc", "raw", "mc_params1_weight.txt", "mc_params1_returns.txt", 4, 200, 0.05, 0.99, 0.01),
    ("gw", "tile", "gw_params1_weight.txt", "gw_params1_returns.txt", 1, 1, 0.0, 1.0, 1.0),
    ("gw", "tile", "gw_params2_weight.txt", "gw_params2_returns.txt", 3, 5, 0.0, 0.9, 0.01),
    ("mc", "tile", "mc_params2_weight.txt", "mc_params2_returns.txt", 25, 200, 0.0, 0.99, 0.005),
]

path_to_ref_dir = os.path.join("reference_output")
ref_files = [
    ("mc_params1_returns.txt", "mc_params1_weight.txt"),
    ("gw_params1_returns.txt", "gw_params1_weight.txt"),
    ("gw_params2_returns.txt", "gw_params2_weight.txt"),
    ("mc_params2_returns.txt", "mc_params2_weight.txt"),
]

assert os.path.isdir(path_to_ref_dir), f"Reference output folder `{path_to_ref_dir}` not found."

print("Generating outputs...")
start = time.time()
for config in configs:
    print(run_command(*config))
time_elapsed = time.time() - start

for (_, _, weight_out, returns_out, *_) in configs:
    assert os.path.isfile(weight_out), f"Weight output file `{weight_out}` not found."
    assert os.path.isfile(returns_out), f"Returns output file `{returns_out}` not found."

print(f"Done generating outputs in {time_elapsed}s.")

eps = 1e-4
class BaseTests(object):
    class BaseCheck(unittest.TestCase):
        idx = None

        def _make_err_msg(self, ref_file, my_file):
            err_msg = "\nThe command for this output was:\n    " + \
                     f"{BASE_PY} {BASE_FILE} " + \
                      " ".join(str(v) for v in configs[self.idx]) + \
                     f"\nThe reference output can be found at {ref_file} " + \
                     f"and your output can be found at {my_file}\n"
            return err_msg

        def test_returns(self):
            ref_return_out = ref_files[self.idx][0]
            ref_return_file = os.path.join(path_to_ref_dir, ref_return_out)
            ref_return = np.genfromtxt(ref_return_file, delimiter="\n")

            my_return_out = configs[self.idx][3]
            my_return = np.genfromtxt(my_return_out, delimiter="\n")

            assert_allclose(my_return, ref_return, atol=eps, 
                            err_msg=self._make_err_msg(ref_return_file, my_return_out))

        def test_weight(self):
            ref_weight_out = ref_files[self.idx][1]
            ref_weight_file = os.path.join(path_to_ref_dir, ref_weight_out)
            ref_weight = np.genfromtxt(ref_weight_file, delimiter=" ")

            my_weight_out = configs[self.idx][2]
            my_weight = np.genfromtxt(my_weight_out, delimiter=" ")

            assert_allclose(my_weight, ref_weight, atol=eps,
                            err_msg=self._make_err_msg(ref_weight_file, my_weight_out))

class MCRaw(BaseTests.BaseCheck):
    idx = 0

class GWSimple(BaseTests.BaseCheck):
    idx = 1

class GW(BaseTests.BaseCheck):
    idx = 2

class MCTile(BaseTests.BaseCheck):
    idx = 3
