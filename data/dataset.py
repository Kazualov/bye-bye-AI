"""Download CORNE-Val from HF and convert it to the benchmark.py layout."""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

KINDS = ["shot", "bg", "mask_sam", "mask_eff"]
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def main(out_dir: Path) -> None:
    src = Path(snapshot_download(repo_id="QinmingZhou/CORNE-Val", repo_type="dataset"))
    root = src / "CORNE-Val" if (src / "CORNE-Val").is_dir() else src

    samples: dict[str, dict[str, Path]] = {}
    for kind in KINDS:
        folder = root / kind
        if not folder.is_dir():
            raise SystemExit(f"Missing folder: {folder}")
        for f in folder.iterdir():
            if f.suffix.lower() in IMG_EXT:
                samples.setdefault(f.stem, {})[kind] = f

    out_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    for sample_id, files in sorted(samples.items()):
        if set(files) != set(KINDS):
            print(f"[skip] {sample_id}: has {sorted(files)}")
            continue
        sample_dir = out_dir / sample_id
        sample_dir.mkdir(exist_ok=True)
        for kind, path in files.items():
            shutil.copy(path, sample_dir / f"{kind}{path.suffix.lower()}")
        kept += 1

    print(f"Wrote {kept} samples to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("data/corne-val"))
    main(p.parse_args().output)