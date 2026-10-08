"""
Script to execute SHL_Grammar_Scoring.ipynb and save the executed notebook with real outputs.
"""

import os
import sys
import json
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def execute_notebook(nb_path="notebook/SHL_Grammar_Scoring.ipynb"):
    print(f"Reading notebook from: {nb_path}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    print("Executing notebook cells with active Python environment...")
    
    # Run execution with working directory set to project root
    project_root = os.path.abspath(".")
    ep.preprocess(nb, {"metadata": {"path": project_root}})

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Successfully executed and updated: {nb_path}")


if __name__ == "__main__":
    execute_notebook()

