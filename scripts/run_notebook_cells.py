"""
Lightweight, robust notebook cell executor using built-in standard library & nbformat.
Executes each cell in order, captures text/display outputs, plots, and writes a fully populated .ipynb file.
"""

import os
import sys
import io
import contextlib
import traceback
import json
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_notebook(nb_path="notebook/SHL_Grammar_Scoring.ipynb"):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    global_env = {
        "__name__": "__main__",
        "display": lambda x: print(x.to_string() if hasattr(x, "to_string") else str(x))
    }

    print(f"Executing cells in {nb_path}...")

    exec_count = 1
    for cell_idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        code_text = "".join(cell["source"])
        # Skip magic commands if any
        clean_lines = [l for l in code_text.split("\n") if not l.strip().startswith("%")]
        clean_code = "\n".join(clean_lines)

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        outputs = []
        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                exec(clean_code, global_env)

            out_str = stdout_capture.getvalue()
            err_str = stderr_capture.getvalue()

            if out_str:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [line + "\n" for line in out_str.splitlines()]
                })
            if err_str:
                outputs.append({
                    "name": "stderr",
                    "output_type": "stream",
                    "text": [line + "\n" for line in err_str.splitlines()]
                })

            cell["execution_count"] = exec_count
            cell["outputs"] = outputs
            exec_count += 1
            print(f"Cell {cell_idx + 1} executed successfully.")

        except Exception as e:
            err_msg = traceback.format_exc()
            print(f"Error in cell {cell_idx + 1}: {e}")
            outputs.append({
                "ename": type(e).__name__,
                "evalue": str(e),
                "output_type": "error",
                "traceback": err_msg.splitlines()
            })
            cell["execution_count"] = exec_count
            cell["outputs"] = outputs
            exec_count += 1

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Executed notebook saved to: {nb_path}")


if __name__ == "__main__":
    run_notebook()

