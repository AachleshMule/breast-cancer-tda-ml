"""
Build a manifest CSV (image_path, label, mask_path, patient_id) for the pipeline.

Modes
  busi    BUSI folder (benign/, malignant/, normal/ with '*_mask*.png' files). Normal images are skipped.
          python scripts/build_manifest.py busi --root /path/Dataset_BUSI_with_GT --out manifests/busi.csv

  folder  A folder with benign/ and malignant/ sub-folders of images (e.g. CBIS-DDSM ROI crops converted to PNG).
          Patient id is parsed from the file name with --patient-regex (first capture group).
          python scripts/build_manifest.py folder --root /path/cbis_roi --out manifests/cbis.csv \
                 --patient-regex "(P_\\d+)"

  csv     Your own table. Map its columns with --image-col/--label-col/--patient-col/--mask-col.
          python scripts/build_manifest.py csv --csv my_table.csv --image-col image --label-col pathology \
                 --patient-col patient_id --out manifests/cbis.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd

EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def busi(root):
    root = Path(root)
    rows = []
    for cls in ("benign", "malignant"):
        for f in sorted((root / cls).iterdir()):
            if f.suffix.lower() not in EXT or "_mask" in f.stem:
                continue
            masks = sorted(p for p in (root / cls).glob(f"{f.stem}_mask*") if p.suffix.lower() in EXT)
            rows.append({"image_path": f"{cls}/{f.name}", "label": cls,
                         "mask_path": ";".join(f"{cls}/{m.name}" for m in masks), "patient_id": ""})
    return pd.DataFrame(rows)


def folder(root, patient_regex=None, mask_suffix=None):
    root = Path(root)
    rows = []
    for cls in ("benign", "malignant"):
        for f in sorted((root / cls).iterdir()):
            if f.suffix.lower() not in EXT or (mask_suffix and f.stem.endswith(mask_suffix)):
                continue
            pid = ""
            if patient_regex:
                m = re.search(patient_regex, f.name)
                pid = m.group(1) if m else ""
            mask = ""
            if mask_suffix:
                cand = f.with_name(f.stem + mask_suffix + f.suffix)
                mask = f"{cls}/{cand.name}" if cand.exists() else ""
            rows.append({"image_path": f"{cls}/{f.name}", "label": cls, "mask_path": mask, "patient_id": pid})
    return pd.DataFrame(rows)


def from_csv(path, image_col, label_col, patient_col=None, mask_col=None):
    df = pd.read_csv(path)
    out = pd.DataFrame({"image_path": df[image_col], "label": df[label_col]})
    out["mask_path"] = df[mask_col] if mask_col else ""
    out["patient_id"] = df[patient_col] if patient_col else ""
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["busi", "folder", "csv"])
    ap.add_argument("--root")
    ap.add_argument("--csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patient-regex")
    ap.add_argument("--mask-suffix")
    ap.add_argument("--image-col")
    ap.add_argument("--label-col")
    ap.add_argument("--patient-col")
    ap.add_argument("--mask-col")
    a = ap.parse_args()
    if a.mode == "busi":
        df = busi(a.root)
    elif a.mode == "folder":
        df = folder(a.root, a.patient_regex, a.mask_suffix)
    else:
        df = from_csv(a.csv, a.image_col, a.label_col, a.patient_col, a.mask_col)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df["label"].value_counts().to_string())
    print(f"wrote {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
