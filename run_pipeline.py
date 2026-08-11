"""Run the complete RPA SLA prediction pipeline in sequence."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
STEPS = [
    "01_generate_data.py",
    "02_explore_clean_preprocess.py",
    "03_cluster_classify_evaluate.py",
]


def main():
    for step in STEPS:
        print(f"\n{'=' * 72}\nRunning {step}\n{'=' * 72}")
        subprocess.run([sys.executable, str(ROOT / step)], cwd=ROOT, check=True)
    print("\nPipeline completed. Review outputs/key_results.csv and plots/.")


if __name__ == "__main__":
    main()
