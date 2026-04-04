import os
import re
import sys


def normalize_label(label):
    s = label.lower()
    s = re.sub(r"\([^)]*\)", "", s)
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s


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
    label_to_ids = {}
    if not os.path.exists(path):
        return id_to_label, label_to_ids
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
                norm = normalize_label(label)
                label_to_ids.setdefault(norm, []).append(idx)
            return id_to_label, label_to_ids


def load_tm_best(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) < 3:
                continue
            try:
                return int(toks[1])
            except ValueError:
                return None
    return None


def load_golden_tm_best(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) < 7:
                continue
            try:
                return int(toks[5])
            except ValueError:
                return None
    return None


def load_geom_status(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        for line in f:
            if line.startswith("STATUS:"):
                return line.strip().split(":", 1)[1].strip()
    return None


def has_golden_boxes(path):
    if not os.path.exists(path):
        return None
    count = 0
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            count += 1
    return count


def main(project_dir):
    expected_path = os.path.join(project_dir, "data", "expected_results.txt")
    mapping_path = os.path.join(project_dir, "data", "template_mapping.txt")
    by_image_dir = os.path.join(project_dir, "results", "by_image")

    expected, skip = parse_expected_results(expected_path)
    if "No_entrance" in skip:
        skip.add("No_enterance")

    id_to_label, label_to_ids = parse_template_mapping(mapping_path)

    keys = sorted(expected.keys())
    lines = []
    lines.append("Failure Summary")
    lines.append("================")

    for key in keys:
        if key in skip:
            continue
        image_dir = os.path.join(by_image_dir, key)
        expected_labels = expected.get(key, [])
        expected_norm = [normalize_label(x) for x in expected_labels]
        expected_ids = []
        for x in expected_norm:
            expected_ids.extend(label_to_ids.get(x, []))
        expect_no_sign = (len(expected_norm) == 1 and expected_norm[0] == "no sign")

        geom_status = load_geom_status(os.path.join(image_dir, "geom_filter_verify.txt"))
        golden_boxes = has_golden_boxes(os.path.join(image_dir, "geom_filtered_golden.txt"))

        golden_id = load_golden_tm_best(os.path.join(image_dir, "template_matching_golden.txt"))
        rtl_id = load_tm_best(os.path.join(image_dir, "actual_template_matching.txt"))

        geom_fail = False
        if expect_no_sign:
            if golden_boxes is not None and golden_boxes > 0:
                geom_fail = True
        else:
            if golden_boxes is not None and golden_boxes == 0:
                geom_fail = True

        tm_fail = False
        if not expect_no_sign:
            if rtl_id is None:
                tm_fail = True
            elif expected_ids and rtl_id not in expected_ids:
                tm_fail = True

        if not geom_fail and not tm_fail:
            continue

        exp_id_str = "none" if not expected_ids else ",".join(str(x) for x in expected_ids)
        exp_label_str = ",".join(expected_labels) if expected_labels else "n/a"
        golden_str = "none" if golden_id is None else f"{golden_id}({id_to_label.get(golden_id, 'id'+str(golden_id))})"
        rtl_str = "none" if rtl_id is None else f"{rtl_id}({id_to_label.get(rtl_id, 'id'+str(rtl_id))})"
        geom_note = f"geom={geom_status}" if geom_status else "geom=unknown"

        reason = []
        if geom_fail:
            reason.append("geom failed")
        if tm_fail:
            reason.append("matching failed")
        reason_str = ", ".join(reason)

        lines.append(
            f"{key}: {reason_str} | expected {exp_label_str} (IDs {exp_id_str}) | golden {golden_str} | rtl {rtl_str} | {geom_note}"
        )

    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("Usage: python summarize_failures.py [project_dir]")
        sys.exit(1)
    proj = sys.argv[1] if len(sys.argv) == 2 else os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    main(proj)
