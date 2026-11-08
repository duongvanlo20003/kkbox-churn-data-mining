"""Download and extract the KKBox competition files into data/.

Needs the Kaggle CLI with a token (~/.kaggle/kaggle.json or ~/.kaggle/access_token) and the
competition rules accepted on kaggle.com. ``user_logs.csv`` is ~7 GB compressed / ~28 GB
extracted; set SKIP_BIG_LOGS=1 to skip it (listening features will then be missing).
Extraction prefers the 7z executable when available (much faster than py7zr).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, RAW_FILES

COMP = "kkbox-churn-prediction-challenge"
SEVEN_ZIP = shutil.which("7z") or next(
    (p for p in [r"C:\Program Files\7-Zip\7z.exe"] if Path(p).exists()), None)


def extract(archive: Path):
    if SEVEN_ZIP:
        subprocess.run([SEVEN_ZIP, "x", "-y", f"-o{DATA}", str(archive)], check=True,
                       stdout=subprocess.DEVNULL)
    else:
        import py7zr
        with py7zr.SevenZipFile(archive) as z:
            z.extractall(DATA)


def main():
    DATA.mkdir(exist_ok=True)
    files = [f for f in RAW_FILES if not (f == "user_logs" and os.environ.get("SKIP_BIG_LOGS"))]
    for name in files:
        if (DATA / f"{name}.csv").exists():
            print("ready:", name, file=sys.stderr)
            continue
        archive = DATA / f"{name}.csv.7z"
        if not archive.exists():
            subprocess.run(["kaggle", "competitions", "download", COMP,
                            "-f", f"{name}.csv.7z", "-p", str(DATA)], check=True)
        extract(archive)
        archive.unlink()
        # some archives nest the csv in a subfolder (e.g. data/churn_comp_refresh/); flatten
        for nested in DATA.rglob(f"{name}.csv"):
            if nested.parent != DATA:
                nested.replace(DATA / nested.name)
        print("ready:", name, file=sys.stderr)


if __name__ == "__main__":
    main()
