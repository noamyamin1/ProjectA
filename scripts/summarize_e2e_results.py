import argparse
import os
import re

SKIP_STEMS = {
    "WhatsApp Image 2026-01-05 at 10.39.09",
    "NO_entrance",
    "No_entrance",
    "No_enterance",
    "slippery_road",
    "down_triangle",
    "10",
    "1",
    "3",
    "4",
    "8",
    "9",
    "26",
    "bumpers",
}


def load_template_mapping(path):
    mapping = {}
    if not os.path.isfile(path):
        return mapping
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^ID\s+(\d+)\s*:\s*.+?\s+-\s+(.+)$", line)
            if match:
                idx = int(match.group(1))
                desc = match.group(2).strip()
                mapping[idx] = desc
    return mapping


def load_expected_results(path):
    expected = {}
    if not os.path.isfile(path):
        return expected
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if " - " not in line:
                continue
            stem, desc = line.split(" - ", 1)
            stem = stem.strip()
            desc = desc.strip()
            expected[stem] = desc
    return expected


def parse_status(path):
    if not os.path.isfile(path):
        return "MISSING"
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip().startswith("STATUS:"):
                return line.split(":", 1)[1].strip()
    return "UNKNOWN"


def parse_geom_filter_counts(path):
    info = {"status": "MISSING", "golden_boxes": None, "rtl_boxes": None}
    if not os.path.isfile(path):
        return info
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if s.startswith("STATUS:"):
                info["status"] = s.split(":", 1)[1].strip()
            elif s.startswith("GOLDEN_BOXES:"):
                try:
                    info["golden_boxes"] = int(s.split(":", 1)[1].strip())
                except Exception:
                    pass
            elif s.startswith("RTL_BOXES:"):
                try:
                    info["rtl_boxes"] = int(s.split(":", 1)[1].strip())
                except Exception:
                    pass
    return info


def parse_actual_class(results_dir, template_map):
    rtl_path = os.path.join(results_dir, "rtl_final_results_e2e.txt")
    if os.path.isfile(rtl_path):
        with open(rtl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip().startswith("CLASS_ID:"):
                    match = re.match(r"CLASS_ID:\s*(\d+)\s*\((.+)\)", line.strip())
                    if match:
                        class_id = int(match.group(1))
                        label = match.group(2).strip()
                        return class_id, label
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        class_id = int(parts[1])
                        label = template_map.get(class_id, "")
                        return class_id, label

    actual_path = os.path.join(results_dir, "actual_template_matching_e2e.txt")
    if os.path.isfile(actual_path):
        with open(actual_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 7:
                    class_id = int(parts[5])
                    label = template_map.get(class_id, "")
                    return class_id, label
    return None, ""


def parse_ids(results_dir):
    rtl_path = os.path.join(results_dir, "rtl_final_results_e2e.txt")
    current_id = None
    last_valid_id = None
    if os.path.isfile(rtl_path):
        with open(rtl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip().startswith("CURRENT_ID:"):
                    try:
                        current_id = int(line.strip().split(":")[1].strip())
                    except Exception:
                        pass
                if line.strip().startswith("LAST_VALID_ID:"):
                    try:
                        last_valid_id = int(line.strip().split(":")[1].strip())
                    except Exception:
                        pass
    return current_id, last_valid_id


def parse_geom_detected(results_dir):
    rtl_path = os.path.join(results_dir, "rtl_final_results_e2e.txt")
    if not os.path.isfile(rtl_path):
        return None
    with open(rtl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip().startswith("GEOM_DETECTED:"):
                try:
                    return int(line.strip().split(":", 1)[1].strip())
                except Exception:
                    return None
    return None


def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def match_expected(actual_label, actual_id, expected_desc):
    if expected_desc is None:
        return "UNKNOWN"

    desc = expected_desc
    if "do not try validating" in desc.lower():
        return "SKIP"

    expected_norm = normalize(desc)
    actual_norm = normalize(actual_label)

    if "no sign" in expected_norm:
        return "PASS" if actual_id in (None, 0) else "FAIL"

    if " or " in expected_norm:
        options = [normalize(item) for item in expected_norm.split(" or ")]
        for opt in options:
            if opt and (opt in actual_norm or actual_norm in opt):
                return "PASS"
        return "FAIL"

    if expected_norm and (expected_norm in actual_norm or actual_norm in expected_norm):
        return "PASS"

    return "FAIL"


def summarize(results_root, expected_path, template_path, out_path):
    template_map = load_template_mapping(template_path)
    expected = load_expected_results(expected_path)
    prev_last_valid_id = None

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("frame,object_detected,template_detected,correct_detection\n")
        for stem in sorted(os.listdir(results_root)):
            if stem in SKIP_STEMS:
                continue
            full = os.path.join(results_root, stem)
            if not os.path.isdir(full):
                continue

            expected_desc = expected.get(stem)
            if expected_desc is None and stem == "No_enterance":
                expected_desc = expected.get("No_entrance")

            geom_info = parse_geom_filter_counts(os.path.join(full, "geom_filter_verify.txt"))
            geom_status = geom_info["status"]
            actual_id, actual_label = parse_actual_class(full, template_map)
            current_id, last_valid_id = parse_ids(full)
            geom_detected = parse_geom_detected(full)
            match_status = match_expected(actual_label, actual_id, expected_desc)

            # Object detected? Use explicit box/detected counters, not verification PASS/FAIL.
            if geom_detected is not None:
                has_object = geom_detected > 0
            elif geom_info["rtl_boxes"] is not None:
                has_object = geom_info["rtl_boxes"] > 0
            else:
                has_object = False
            object_detected = "YES" if has_object else "NO"

            # Template detected?
            if object_detected == "YES" and actual_label:
                template_detected = actual_label
            else:
                template_detected = "NO"

            # Correct detection?
            if object_detected == "NO":
                # Validate no-bbox behavior: current_id==255 and last_valid_id unchanged.
                no_sign_ok = (current_id == 255)
                last_valid_unchanged = (prev_last_valid_id is None or last_valid_id == prev_last_valid_id)
                if no_sign_ok and last_valid_unchanged:
                    correct_detection = "NO OBJECT"
                else:
                    correct_detection = "NO_BBOX_WRONG"
            elif match_status == "PASS":
                correct_detection = "CORRECT"
            elif match_status == "FAIL":
                correct_detection = "WRONG"
            else:
                correct_detection = match_status

            handle.write(f"{stem},{object_detected},{template_detected},{correct_detection}\n")
            if last_valid_id is not None:
                prev_last_valid_id = last_valid_id


def main():
    parser = argparse.ArgumentParser(description="Summarize per-frame E2E results.")
    parser.add_argument("--project-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    project_root = args.project_root
    results_root = os.path.join(project_root, "results", "by_image")
    expected_path = os.path.join(project_root, "data", "expected_results.txt")
    template_path = os.path.join(project_root, "data", "template_mapping.txt")
    out_path = args.out or os.path.join(project_root, "results", "e2e_frame_summary.txt")

    summarize(results_root, expected_path, template_path, out_path)


if __name__ == "__main__":
    main()
