"""Give structure masks that were saved without geometry the affine of their CT.

Some GTVp masks were written with an identity affine (qform_code 0, sform_code 2) but with the
correct voxel sizes and the same voxel grid as the CT. This script copies the CT's qform/sform
onto those masks. The voxel data is left untouched.

Only masks whose affine is the identity are changed; everything else is reported and skipped.
Originals are copied to a backup directory before they are overwritten, and a CSV log of every
case is written next to the backups.

Dry run (default, nothing is written):
    uv run python scripts/fix_gtv_affine.py ROOT --cases msp_pipeline/results/failures.csv

Apply:
    uv run python scripts/fix_gtv_affine.py ROOT --cases msp_pipeline/results/failures.csv --apply
"""

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

ATOL = 1e-3


def read_cases(values: list[str]) -> list[str]:
    """Case ids given directly, or read from a .csv (column 'case_id') or .txt (one per line)."""
    cases = []
    for value in values:
        path = Path(value)
        if path.suffix == ".csv" and path.is_file():
            with path.open() as f:
                cases += [row["case_id"] for row in csv.DictReader(f)]
        elif path.suffix == ".txt" and path.is_file():
            cases += [line.strip() for line in path.read_text().splitlines() if line.strip()]
        else:
            cases.append(value)
    return list(dict.fromkeys(cases))  # de-duplicate, keep order


def find_nifti(case_dir: Path, name: str) -> Path:
    for suffix in (".nii.gz", ".nii"):
        path = case_dir / f"{name}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"no {name}.nii(.gz) in {case_dir}")


def fixed_image(mask: nib.Nifti1Image, ct: nib.Nifti1Image) -> nib.Nifti1Image:
    """The mask's voxel data and header, with the CT's qform and sform."""
    header = mask.header.copy()
    header.set_qform(ct.header.get_qform(), code=int(ct.header["qform_code"]))
    header.set_sform(ct.header.get_sform(), code=int(ct.header["sform_code"]))
    return nib.Nifti1Image(np.asanyarray(mask.dataobj), ct.affine, header=header)


def process_case(
    case_dir: Path, image: str, structure: str
) -> tuple[str, str, Path | None, object]:
    """Return (status, detail, mask path, fixed image or None)."""
    ct_path = find_nifti(case_dir, image)
    mask_path = find_nifti(case_dir, structure)
    ct = nib.load(ct_path)
    mask = nib.load(mask_path)

    if mask.shape[:3] != ct.shape[:3]:
        return "skipped", f"shape {mask.shape} != CT {ct.shape}", mask_path, None
    if np.allclose(mask.affine, ct.affine, atol=ATOL):
        return "ok", "affine already matches the CT", mask_path, None
    if not np.allclose(mask.affine, np.eye(4), atol=ATOL):
        return "skipped", "affine differs from the CT but is not the identity", mask_path, None
    mask_zooms, ct_zooms = mask.header.get_zooms()[:3], ct.header.get_zooms()[:3]
    if not np.allclose(mask_zooms, ct_zooms, atol=ATOL):
        return "skipped", f"voxel sizes {mask_zooms} != CT {ct_zooms}", mask_path, None

    return "fix", "identity affine replaced by the CT affine", mask_path, fixed_image(mask, ct)


def verify(written: Path, original_data: np.ndarray, ct: nib.Nifti1Image) -> None:
    img = nib.load(written)
    if not np.allclose(img.affine, ct.affine, atol=ATOL):
        raise RuntimeError(f"{written}: affine still differs from the CT after writing")
    if not np.array_equal(np.asanyarray(img.dataobj), original_data):
        raise RuntimeError(f"{written}: voxel data changed while writing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="dataset root, one sub-directory per case")
    parser.add_argument(
        "--cases",
        nargs="+",
        required=True,
        help="case ids, or a .csv with a 'case_id' column (e.g. failures.csv), or a .txt file",
    )
    parser.add_argument("--structure", default="GTVp", help="mask file name without extension")
    parser.add_argument("--image", default="image", help="CT file name without extension")
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="where originals and the log go (default: <root>_affine_backup next to root)",
    )
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = parser.parse_args()

    root = args.root.resolve()
    backup_dir = (args.backup_dir or root.parent / f"{root.name}_affine_backup").resolve()
    cases = read_cases(args.cases)
    print(f"{'APPLY' if args.apply else 'DRY RUN'}: {len(cases)} case(s) in {root}")

    log = []
    for case_id in cases:
        try:
            status, detail, mask_path, fixed = process_case(
                root / case_id, args.image, args.structure
            )
            if status == "fix" and args.apply:
                backup = backup_dir / case_id / mask_path.name
                backup.parent.mkdir(parents=True, exist_ok=True)
                if backup.exists():
                    raise FileExistsError(f"backup {backup} already exists, not overwriting it")
                shutil.copy2(mask_path, backup)
                original_data = np.asanyarray(fixed.dataobj)
                nib.save(fixed, mask_path)
                verify(mask_path, original_data, nib.load(find_nifti(root / case_id, args.image)))
                status = "fixed"
        except Exception as e:  # noqa: BLE001 - report and continue with the next case
            status, detail = "error", f"{type(e).__name__}: {e}"
        print(f"  {case_id:15s} {status:8s} {detail}")
        log.append(
            {"case_id": case_id, "structure": args.structure, "status": status, "detail": detail}
        )

    counts = {
        s: sum(r["status"] == s for r in log) for s in dict.fromkeys(r["status"] for r in log)
    }
    print("summary:", ", ".join(f"{n} {s}" for s, n in counts.items()))

    if args.apply:
        backup_dir.mkdir(parents=True, exist_ok=True)
        log_path = backup_dir / f"fix_affine_{datetime.now().astimezone():%Y%m%d_%H%M%S}.csv"
        with log_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case_id", "structure", "status", "detail"])
            writer.writeheader()
            writer.writerows(log)
        print(f"log and backups: {backup_dir}")
    else:
        print("nothing written; rerun with --apply to fix the 'fix' cases")


if __name__ == "__main__":
    main()
