from __future__ import annotations

import argparse
from pathlib import Path

from src.io_utils import load_config
from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run multi-site memory kernel recovery pipeline.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    result_df = run_pipeline(config)
    print("\nPipeline completed successfully.")
    print(f"Summary table saved to: {Path(config['output_dir']) / 'cross_site_kernel_fit_summary.csv'}")
    print("\nTop rows:")
    print(result_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
