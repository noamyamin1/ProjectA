import sys
import os


def load_stats(path):
    data = {}
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) < 7:
                continue
            label = int(toks[0])
            data[label] = {
                "area": int(toks[1]),
                "perimeter": int(toks[2]),
                "xmin": int(toks[3]),
                "xmax": int(toks[4]),
                "ymin": int(toks[5]),
                "ymax": int(toks[6]),
            }
    return data


def main(golden_path, rtl_path, out_dir):
    golden = load_stats(golden_path)
    rtl = load_stats(rtl_path)

    labels = sorted(set(golden.keys()) | set(rtl.keys()))

    missing_in_rtl = []
    extra_in_rtl = []
    mismatches = []

    for lbl in labels:
        g = golden.get(lbl)
        r = rtl.get(lbl)
        if g is None:
            extra_in_rtl.append(lbl)
            continue
        if r is None:
            missing_in_rtl.append(lbl)
            continue

        for key in ["area", "perimeter", "xmin", "xmax", "ymin", "ymax"]:
            if g[key] != r[key]:
                mismatches.append((lbl, key, g[key], r[key]))

    status = "PASS" if (not missing_in_rtl and not extra_in_rtl and not mismatches) else "FAIL"

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ccl_stats_verify.txt")
    with open(out_path, "w") as f:
        f.write("STAGE: CCL_STATS\n")
        f.write(f"GOLDEN_LABELS: {len(golden)}\n")
        f.write(f"RTL_LABELS: {len(rtl)}\n")
        f.write(f"MISSING_LABELS_IN_RTL: {len(missing_in_rtl)}\n")
        f.write(f"EXTRA_LABELS_IN_RTL: {len(extra_in_rtl)}\n")
        f.write(f"FIELD_MISMATCHES: {len(mismatches)}\n")
        f.write(f"STATUS: {status}\n")

        if missing_in_rtl:
            f.write("\nMissing labels in RTL:\n")
            for lbl in missing_in_rtl[:200]:
                f.write(f"{lbl}\n")

        if extra_in_rtl:
            f.write("\nExtra labels in RTL:\n")
            for lbl in extra_in_rtl[:200]:
                f.write(f"{lbl}\n")

        if mismatches:
            f.write("\nField mismatches (label field golden rtl):\n")
            for lbl, key, gv, rv in mismatches[:500]:
                f.write(f"{lbl} {key} {gv} {rv}\n")

    print("----------------------------------------")
    print("CCL STATS VERIFICATION")
    print("----------------------------------------")
    print(f"Golden labels: {len(golden)}")
    print(f"RTL labels: {len(rtl)}")
    print(f"Missing labels in RTL: {len(missing_in_rtl)}")
    print(f"Extra labels in RTL: {len(extra_in_rtl)}")
    print(f"Field mismatches: {len(mismatches)}")
    print(f"Overall Status: {status}")
    print(f"Report: {out_path}")

    if status != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python verify_ccl_stats.py <ccl_stats_golden.txt> <actual_ccl_stats.txt> <out_results_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
