import os
import re
import sys

NO_SIGN_SCORE_THRESHOLD = 230


def parse_expected_results(path):
    expected = {}
    skip = set()
    if not os.path.exists(path):
        return expected, skip
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if " - " not in s:
                continue
            key, rest = s.split(" - ", 1)
            key = key.strip()
            rest = rest.strip()
            if "do not try validating" in rest:
                skip.add(key)
            rest = re.sub(r"\([^)]*\)", "", rest).strip()
            if not rest:
                continue
            expected[key] = [r.strip() for r in rest.split(" or ") if r.strip()]
    return expected, skip


def parse_template_mapping(path):
    id_to_label = {}
    if not os.path.exists(path):
        return id_to_label
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if not s.startswith("ID") or ":" not in s:
                continue
            left, right = s.split(":", 1)
            left_toks = left.strip().split()
            if len(left_toks) < 2:
                continue
            try:
                idx = int(left_toks[1])
            except ValueError:
                continue
            right = right.strip()
            if not right:
                continue
            _fname, sep, label = right.partition(" - ")
            label = label.strip() if sep else _fname.strip()
            id_to_label[idx] = label
    return id_to_label


def normalize_label(label):
    s = label.lower()
    s = re.sub(r"\([^)]*\)", "", s)
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s


def load_actual_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) < 3:
                continue
            try:
                roi_id = int(toks[0])
                best_id = int(toks[1])
                best_score = int(toks[2])
            except ValueError:
                continue
            rows.append((roi_id, best_id, best_score))
    return rows


def count_roi_list(path):
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            count += 1
    return count


def select_best_actual(rows):
    if not rows:
        return None
    best = rows[0]
    for row in rows[1:]:
        if row[2] < best[2]:
            best = row
    return best


def main(project_dir, out_path=None):
    expected_path = os.path.join(project_dir, "data", "expected_results.txt")
    mapping_path = os.path.join(project_dir, "data", "template_mapping.txt")
    results_root = os.path.join(project_dir, "results", "by_image")

    expected, skip = parse_expected_results(expected_path)
    if "No_entrance" in skip:
        skip.add("No_enterance")
    id_to_label = parse_template_mapping(mapping_path)

    if out_path is None:
        out_path = os.path.join(project_dir, "results", "expected_vs_actual_report.txt")

    expected_keys = list(expected.keys())
    expected_keys.sort()

    known_dirs = set()
    if os.path.isdir(results_root):
        for name in os.listdir(results_root):
            full = os.path.join(results_root, name)
            if os.path.isdir(full):
                known_dirs.add(name)

    total = 0
    passed = 0
    failed = 0
    skipped = 0
    missing = 0

    lines = []
    lines.append("Expected vs Actual Template Matching")
    lines.append("=====================================")
    lines.append("")

    for key in expected_keys:
        if key in skip:
            skipped += 1
            continue
        total += 1
        image_dir = os.path.join(results_root, key)
        expected_labels = expected.get(key, [])
        expected_norm = {normalize_label(x) for x in expected_labels}
        expect_no_sign = (len(expected_norm) == 1 and "no sign" in expected_norm)

        if not os.path.isdir(image_dir):
            missing += 1
            lines.append(f"{key}: MISSING_RESULTS (expected: {', '.join(expected_labels)})")
            continue

        actual_path = os.path.join(image_dir, "actual_template_matching.txt")
        roi_list_path = os.path.join(image_dir, "roi_list.txt")
        rows = load_actual_rows(actual_path)
        roi_count = len(rows)
        if roi_count == 0:
            roi_count = count_roi_list(roi_list_path)

        if expect_no_sign:
            if roi_count == 0:
                passed += 1
                lines.append(f"{key}: PASS (expected no sign, no ROIs)")
            else:
                best = select_best_actual(rows)
                if best is not None and best[2] >= NO_SIGN_SCORE_THRESHOLD:
                    passed += 1
                    lines.append(
                        f"{key}: PASS (expected no sign, best score={best[2]} >= {NO_SIGN_SCORE_THRESHOLD})"
                    )
                else:
                    failed += 1
                    lines.append(f"{key}: FAIL (expected no sign, got {roi_count} ROI(s))")
            continue

        if roi_count == 0:
            failed += 1
            lines.append(f"{key}: FAIL (expected {', '.join(expected_labels)}, got no ROIs)")
            continue

        best = select_best_actual(rows)
        if best is None:
            failed += 1
            lines.append(f"{key}: FAIL (expected {', '.join(expected_labels)}, missing actual results)")
            continue

        roi_id, best_id, best_score = best
        if best_score >= NO_SIGN_SCORE_THRESHOLD:
            failed += 1
            lines.append(
                f"{key}: FAIL (expected {', '.join(expected_labels)}, got no sign score={best_score})"
            )
            continue
        pred_label = id_to_label.get(best_id, f"id{best_id}")
        pred_norm = normalize_label(pred_label)
        if pred_norm in expected_norm:
            passed += 1
            lines.append(f"{key}: PASS (expected {', '.join(expected_labels)}, got {best_id} {pred_label} score={best_score})")
        else:
            failed += 1
            lines.append(f"{key}: FAIL (expected {', '.join(expected_labels)}, got {best_id} {pred_label} score={best_score})")

    extra_dirs = sorted([d for d in known_dirs if d not in expected and d not in skip])

    lines.append("")
    lines.append("Summary")
    lines.append("-------")
    lines.append(f"Total checked: {total}")
    lines.append(f"Passed: {passed}")
    lines.append(f"Failed: {failed}")
    lines.append(f"Missing results: {missing}")
    lines.append(f"Skipped: {skipped}")
    if extra_dirs:
        lines.append("")
        lines.append("Unexpected result directories:")
        for d in extra_dirs:
            lines.append(f"- {d}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report: {out_path}")
    print(f"Checked: {total} | Passed: {passed} | Failed: {failed} | Missing: {missing} | Skipped: {skipped}")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("Usage: python report_expected_results.py [project_dir]")
        sys.exit(1)
    project_dir = sys.argv[1] if len(sys.argv) == 2 else os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    main(project_dir)
