# create_sequential_mixed.py
# Creates a sequential mixed CSV: first N normal beats, then M arrhythmic beats.
# Usage example:
#   python3 create_sequential_mixed.py
# (edit the variables below or pass CLI args if you want)

import pandas as pd
import argparse
import sys

def main():
    p = argparse.ArgumentParser(description="Build sequential mixed CSV (normal -> arrhythmia).")
    p.add_argument("--normal", default="normal_segment.csv", help="Path to normal CSV")
    p.add_argument("--arr", default="arrhythmia_segment.csv", help="Path to arrhythmia CSV")
    p.add_argument("--out", default="mixed_sequential.csv", help="Output CSV path")
    p.add_argument("--n_normal", type=int, default=100, help="Number of normal beats at start")
    p.add_argument("--n_arr", type=int, default=200, help="Number of arrhythmic beats after normal")
    p.add_argument("--preserve_order", action="store_true",
                   help="If set, keep original file order; otherwise sample from top")
    args = p.parse_args()

    try:
        df_norm = pd.read_csv(args.normal)
    except Exception as e:
        print(f"ERROR reading normal file '{args.normal}': {e}", file=sys.stderr)
        sys.exit(1)

    try:
        df_arr = pd.read_csv(args.arr)
    except Exception as e:
        print(f"ERROR reading arrhythmia file '{args.arr}': {e}", file=sys.stderr)
        sys.exit(1)

    # align columns (use common cols only)
    common_cols = [c for c in df_norm.columns if c in df_arr.columns]
    if len(common_cols) == 0:
        print("No common columns between the two CSVs — cannot merge.", file=sys.stderr)
        sys.exit(1)

    df_norm = df_norm[common_cols].copy()
    df_arr = df_arr[common_cols].copy()

    # choose rows
    if args.preserve_order:
        n_norm = min(args.n_normal, len(df_norm))
        n_arr = min(args.n_arr, len(df_arr))
        part_norm = df_norm.iloc[:n_norm].copy()
        part_arr = df_arr.iloc[:n_arr].copy()
    else:
        # sample without replacement from top if available, otherwise allow replacement
        n_norm = args.n_normal
        n_arr = args.n_arr
        replace_norm = n_norm > len(df_norm)
        replace_arr = n_arr > len(df_arr)
        part_norm = df_norm.sample(n=n_norm, replace=replace_norm, random_state=42).reset_index(drop=True)
        part_arr = df_arr.sample(n=n_arr, replace=replace_arr, random_state=42).reset_index(drop=True)

    # option: add a small separator (optional)
    # combined = pd.concat([part_norm, part_arr], axis=0, ignore_index=True)
    combined = pd.concat([part_norm, part_arr], axis=0, ignore_index=True)

    combined.to_csv(args.out, index=False)
    print(f"Saved mixed sequential file to: {args.out} (normal={len(part_norm)}, arrhythmia={len(part_arr)})")

if __name__ == "__main__":
    main()
