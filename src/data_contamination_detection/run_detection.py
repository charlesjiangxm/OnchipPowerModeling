#!/usr/bin/env python3
"""Data contamination detection for on-chip power modeling.

Implements three detection methods per spec.md:
  1. Sample-level nearest-neighbor (faiss ANN + baseline comparison)
  2. Segment-level (KMeans tokenization + shingling + MinHash LSH)
  3. Adversarial validation (LightGBM classifier)

Usage:
  python3 run_detection.py                    # full run
  python3 run_detection.py --smoke-test       # quick test (1000 rows/file)
  python3 run_detection.py --methods 1 3      # run only methods 1 and 3
  python3 run_detection.py --stumpy           # enable stumpy matrix profile
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import data_loader
import preprocess
import method1_ann
import method2_segment
import method3_adversarial
import reporting


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "run.log"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)


def main():
    parser = argparse.ArgumentParser(description="Data contamination detection")
    parser.add_argument("--data-dir", default=None, help="Path to aq_core directory with PKL files")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "output"), help="Output directory")
    parser.add_argument("--methods", nargs="+", default=["1", "2", "3"], choices=["1", "2", "3"],
                        help="Which methods to run")
    parser.add_argument("--smoke-test", action="store_true", help="Use first 1000 rows per file")
    parser.add_argument("--stumpy", action="store_true", help="Enable stumpy matrix profile in method 2")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    logger = logging.getLogger("run_detection")

    logger.info("=" * 60)
    logger.info("Data Contamination Detection Pipeline")
    logger.info(f"Methods: {args.methods}, smoke_test={args.smoke_test}, stumpy={args.stumpy}")
    logger.info("=" * 60)

    # Step 1: Load data
    t0 = time.time()
    train_data, test_data = data_loader.load_data(args.data_dir, args.smoke_test)
    logger.info(f"Data loaded in {time.time()-t0:.1f}s")
    logger.info(f"  train: {train_data.X.shape}, test: {test_data.X.shape}")

    # Step 2: Preprocess
    t0 = time.time()
    pp = preprocess.run(train_data, test_data)
    logger.info(f"Preprocessing done in {time.time()-t0:.1f}s")

    # Prepare metadata
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": pp.X_train.shape[0],
        "n_test": pp.X_test.shape[0],
        "n_features_raw": len(pp.keep_mask),
        "n_features_after_drop": int(pp.keep_mask.sum()),
        "n_constant_dropped": int((~pp.keep_mask).sum()),
        "train_files": [b.name for b in train_data.boundaries],
        "test_file": test_data.boundaries[0].name if test_data.boundaries else "coremark_func.pkl",
        "seed": 42,
        "smoke_test": args.smoke_test,
    }

    m1_result = None
    m2_result = None
    m3_result = None

    # Step 3: Run methods
    if "1" in args.methods:
        logger.info("-" * 40)
        logger.info("Running Method 1: Sample-Level Neighbors")
        logger.info("-" * 40)
        try:
            t0 = time.time()
            m1_result = method1_ann.run(
                pp.X_train, pp.X_test,
                train_data.source_ids, train_data.boundaries,
            )
            logger.info(f"Method 1 done in {time.time()-t0:.1f}s")
        except Exception as e:
            logger.error(f"Method 1 failed: {e}", exc_info=True)

    if "2" in args.methods:
        logger.info("-" * 40)
        logger.info("Running Method 2: Segment-Level (MinHash LSH)")
        logger.info("-" * 40)
        try:
            t0 = time.time()
            m2_result = method2_segment.run(
                pp.X_train, pp.X_test,
                train_data.source_ids, train_data.boundaries,
                test_data.source_ids, test_data.boundaries,
                enable_stumpy=args.stumpy,
            )
            logger.info(f"Method 2 done in {time.time()-t0:.1f}s")
        except Exception as e:
            logger.error(f"Method 2 failed: {e}", exc_info=True)

    if "3" in args.methods:
        logger.info("-" * 40)
        logger.info("Running Method 3: Adversarial Validation")
        logger.info("-" * 40)
        try:
            t0 = time.time()
            m3_result = method3_adversarial.run(pp.X_train, pp.X_test, pp.feature_names)
            logger.info(f"Method 3 done in {time.time()-t0:.1f}s")
        except Exception as e:
            logger.error(f"Method 3 failed: {e}", exc_info=True)

    # Step 4: Generate report
    logger.info("-" * 40)
    logger.info("Generating report...")
    logger.info("-" * 40)
    metadata["_qt"] = pp.qt
    if args.no_plots:
        # Still save metrics JSON and raw arrays
        reporting._ensure_dirs(output_dir)
        reporting.save_raw(output_dir, m1_result, m2_result, pp.qt)
        metadata.pop("_qt", None)
        metrics = reporting.build_metrics_json(metadata, m1_result, m2_result, m3_result)
        import json
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)
    else:
        reporting.generate_report(output_dir, metadata, m1_result, m2_result, m3_result)

    logger.info("Pipeline complete. Output at: " + str(output_dir))


if __name__ == "__main__":
    main()
