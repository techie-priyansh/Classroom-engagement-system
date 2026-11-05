"""Project paths, so no script needs a path from one person's computer.

PROJECT_ROOT   the repository folder (found from this file, works from any
               working directory)
DATASET_DIR    where PTZ recordings are saved
CALIBRATION_DIR  camera calibration images and results. By default the
               folder `calibration` NEXT TO the repository; set the
               environment variable MTP_CALIBRATION_DIR to use another one.
"""

import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

CALIBRATION_DIR = Path(
    os.environ.get("MTP_CALIBRATION_DIR", PROJECT_ROOT.parent / "calibration")
)

TEST_VIDEO = PROJECT_ROOT / "tests" / "NR131_test1_30Sec.mp4"
RESULTS_DIR = PROJECT_ROOT / "results"
