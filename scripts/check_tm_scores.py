#!/usr/bin/env python3
import argparse
import os


def load_hex_rows(path):
    rows = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(int(s, 16))
    return rows


def load_templates_mem(path):
    rows = load_hex_rows(path)
    if len(rows) < 32:
        raise RuntimeError("templates.mem has fewer than 32 rows")
    if len(rows) % 32 != 0:
        raise RuntimeError("templates.mem row count is not a multiple of 32")
    templates = []
    for i in range(0, len(rows), 32):
        templates.append(rows[i:i + 32])
    return templates


def parse_mapping(path, count):
    id_to_label = {i: f"id{i}" for i in range(count)}
    if not path or not os.path.exists(path):
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
            if 0 <= idx < count:
                id_to_label[idx] = label
    return id_to_label


def popcount32(x):
    return bin(int(x) & 0xFFFFFFFF).count("1")


def score_template_rtl(bin_rows, tmpl_rows):
    best = None
    best_dx = 0
    best_dy = 0
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            mismatches = 0
            for row_idx in range(32):
                actual_y = row_idx + dy
                if actual_y < 0 or actual_y >= 32:
                    continue
                raw_row = bin_rows[actual_y]
                if dx > 0:
                    shifted = (raw_row << dx) & 0xFFFFFFFF
                    mask = (0xFFFFFFFF << dx) & 0xFFFFFFFF
                elif dx < 0:
                    shifted = (raw_row >> (-dx)) & 0xFFFFFFFF
                    mask = (0xFFFFFFFF >> (-dx)) & 0xFFFFFFFF
                else:
                    shifted = raw_row
                    mask = 0xFFFFFFFF
                xor_val = (shifted ^ tmpl_rows[row_idx]) & mask
                mismatches += popcount32(xor_val)
            if best is None or mismatches < best:
                best = mismatches
                best_dx = dx
                best_dy = dy
    return best if best is not None else 0, best_dx, best_dy


def score_template_rtl_pipeline(bin_rows, tmpl_rows):
    best = None
    best_dx = 0
    best_dy = 0

    for dy_idx in range(9):
        for dx_idx in range(9):
            actual_x = dx_idx - 4
            current_mismatches = 0
            template_ram_addr = 0
            pl_raw_roi_row = 0
            pl_shifted_roi_row = 0
            pl_rom_data = 0
            pl_valid_mask = 0
            pl_eval_valid_1 = 0
            pl_eval_valid_2 = 0
            eval_valid = 0

            for match_row_cnt in range(34):
                actual_y = match_row_cnt + dy_idx - 4
                if 0 <= actual_y < 32 and match_row_cnt < 32:
                    raw_row = bin_rows[actual_y]
                else:
                    raw_row = 0

                if actual_x > 0:
                    current_x_mask = (0xFFFFFFFF << actual_x) & 0xFFFFFFFF
                elif actual_x < 0:
                    current_x_mask = (0xFFFFFFFF >> (-actual_x)) & 0xFFFFFFFF
                else:
                    current_x_mask = 0xFFFFFFFF

                if actual_y < 0 or actual_y >= 32 or match_row_cnt >= 32:
                    current_x_mask = 0

                xor_val = (pl_shifted_roi_row ^ pl_rom_data) & pl_valid_mask
                if eval_valid:
                    current_mismatches += popcount32(xor_val)

                if actual_x > 0:
                    next_shifted = (pl_raw_roi_row << actual_x) & 0xFFFFFFFF
                elif actual_x < 0:
                    next_shifted = (pl_raw_roi_row >> (-actual_x)) & 0xFFFFFFFF
                else:
                    next_shifted = pl_raw_roi_row & 0xFFFFFFFF

                next_pl_raw = raw_row & 0xFFFFFFFF
                next_pl_shifted = next_shifted
                next_pl_rom = tmpl_rows[template_ram_addr]
                next_pl_valid_mask = current_x_mask & 0xFFFFFFFF
                next_pl_eval_valid_1 = 1 if match_row_cnt < 32 else 0
                next_pl_eval_valid_2 = pl_eval_valid_1
                next_eval_valid = pl_eval_valid_2

                next_template_ram_addr = template_ram_addr
                if match_row_cnt < 32:
                    next_template_ram_addr = match_row_cnt

                pl_raw_roi_row = next_pl_raw
                pl_shifted_roi_row = next_pl_shifted
                pl_rom_data = next_pl_rom
                pl_valid_mask = next_pl_valid_mask
                pl_eval_valid_1 = next_pl_eval_valid_1
                pl_eval_valid_2 = next_pl_eval_valid_2
                eval_valid = next_eval_valid
                template_ram_addr = next_template_ram_addr

            if best is None or current_mismatches < best:
                best = current_mismatches
                best_dx = dx_idx - 4
                best_dy = dy_idx - 4

    return best if best is not None else 0, best_dx, best_dy


def load_actual_scores(path, count):
    if not path or not os.path.exists(path):
        return None
    scores = [None] * count
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            toks = s.split()
            if len(toks) < 2:
                continue
            try:
                tid = int(toks[0])
                sc = int(toks[1])
            except ValueError:
                continue
            if 0 <= tid < count:
                scores[tid] = sc
    if any(v is None for v in scores):
        return None
    return scores


def main():
    parser = argparse.ArgumentParser(description="Check template matching scores against RTL-style scoring")
    parser.add_argument("roi_bin", help="Path to ROI bin rows (32 hex rows)")
    parser.add_argument("--templates-mem", default="data/templates.mem", help="Path to templates.mem")
    parser.add_argument("--mapping", default="data/template_mapping.txt", help="Path to template mapping")
    parser.add_argument("--actual-scores", default=None, help="Path to actual_scores_full.txt")
    parser.add_argument("--top", type=int, default=3, help="Top-N to report")
    parser.add_argument("--dump-all", action="store_true", help="Print all template scores")
    parser.add_argument("--mode", choices=["ideal", "pipeline"], default="ideal",
                        help="Scoring mode: ideal (row-aligned) or pipeline (RTL timing)")
    args = parser.parse_args()

    bin_rows = load_hex_rows(args.roi_bin)
    if len(bin_rows) < 32:
        raise RuntimeError("ROI bin has fewer than 32 rows")
    bin_rows = bin_rows[:32]

    templates = load_templates_mem(args.templates_mem)
    id_to_label = parse_mapping(args.mapping, len(templates))
    actual_scores = load_actual_scores(args.actual_scores, len(templates))

    scores = []
    shifts = []
    for tidx, tmpl_rows in enumerate(templates):
        if args.mode == "pipeline":
            sc, dx, dy = score_template_rtl_pipeline(bin_rows, tmpl_rows)
        else:
            sc, dx, dy = score_template_rtl(bin_rows, tmpl_rows)
        scores.append(sc)
        shifts.append((dx, dy))

    ranked = sorted(enumerate(scores), key=lambda x: x[1])
    topn = ranked[: max(args.top, 1)]

    print("Computed top{} (mode={}):".format(args.top, args.mode))
    for tid, sc in topn:
        dx, dy = shifts[tid]
        print("  {} {} score={} dx={} dy={}".format(tid, id_to_label.get(tid, f"id{tid}"), sc, dx, dy))

    if actual_scores is not None:
        actual_ranked = sorted(enumerate(actual_scores), key=lambda x: x[1])
        actual_top = actual_ranked[: max(args.top, 1)]
        print("Actual top{}:".format(args.top))
        for tid, sc in actual_top:
            print("  {} {} score={}".format(tid, id_to_label.get(tid, f"id{tid}"), sc))

        mismatches = []
        for tid, sc in enumerate(scores):
            if sc != actual_scores[tid]:
                mismatches.append((tid, sc, actual_scores[tid]))

        print("Score mismatches: {} of {}".format(len(mismatches), len(scores)))
        for tid, sc, act in mismatches[:10]:
            print("  {} {} computed={} actual={}".format(tid, id_to_label.get(tid, f"id{tid}"), sc, act))

    if args.dump_all:
        print("\nAll template scores:")
        for tid, sc in enumerate(scores):
            label = id_to_label.get(tid, f"id{tid}")
            dx, dy = shifts[tid]
            print("{} {} score={} dx={} dy={}".format(tid, label, sc, dx, dy))


if __name__ == "__main__":
    main()
