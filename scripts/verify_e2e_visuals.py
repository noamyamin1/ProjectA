import argparse
import os

from verify_red_mask import main as verify_red_mask
from verify_morphology import main as verify_morphology
from verify_ccl import main as verify_ccl
from verify_geom_filter import main as verify_geom_filter
from verify_roi_template import main as verify_roi_template


def find_image_path(stem, image_dir):
	for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
		path = os.path.join(image_dir, f"{stem}{ext}")
		if os.path.exists(path):
			return path
	return None


def choose_template_dir(project_root, explicit=None):
	if explicit:
		return explicit
	cand = os.path.join(project_root, "pyton", "Templates_white")
	if os.path.isdir(cand):
		return cand
	return os.path.join(project_root, "pyton", "Templates")


def list_e2e_stems(out_root):
	stems = []
	if not os.path.isdir(out_root):
		return stems
	for name in sorted(os.listdir(out_root)):
		full = os.path.join(out_root, name)
		if not os.path.isdir(full):
			continue
		if os.path.exists(os.path.join(full, "actual_mask_out_e2e.txt")):
			stems.append(name)
	return stems


def should_skip_stem(stem):
	skip_stems = {
		"WhatsApp Image 2026-01-05 at 10.39.09",
		"NO_entrance",
		"No_entrance",
		"No_enterance",
		"slippery_road",
		"down_triangle",
		"1",
		"3",
		"4",
		"9",
		"26",
	}
	return stem in skip_stems


def run_for_stem(stem, golden_root, out_root, image_dir, template_dir):
	golden_dir = os.path.join(golden_root, stem)
	out_dir = os.path.join(out_root, stem)

	golden_mask = os.path.join(golden_dir, "mask_out.txt")
	actual_mask = os.path.join(out_dir, "actual_mask_out_e2e.txt")
	if os.path.exists(golden_mask) and os.path.exists(actual_mask):
		verify_red_mask(golden_mask, actual_mask, out_dir)

	golden_morph = os.path.join(golden_dir, "morph_out.txt")
	actual_morph = os.path.join(out_dir, "actual_morph_out_e2e.txt")
	if os.path.exists(golden_morph) and os.path.exists(actual_morph):
		verify_morphology(golden_morph, actual_morph, out_dir)

	golden_ccl_p1 = os.path.join(golden_dir, "ccl_pass1_golden.txt")
	golden_ccl_p2 = os.path.join(golden_dir, "ccl_pass2_golden.txt")
	actual_ccl_p1 = os.path.join(out_dir, "actual_ccl_pass1_e2e.txt")
	actual_ccl_p2 = os.path.join(out_dir, "actual_ccl_pass2_e2e.txt")
	if (os.path.exists(golden_ccl_p1) and os.path.exists(golden_ccl_p2) and
			os.path.exists(actual_ccl_p1) and os.path.exists(actual_ccl_p2)):
		verify_ccl(golden_ccl_p1, actual_ccl_p1, golden_ccl_p2, actual_ccl_p2, out_dir)

	golden_geom = os.path.join(golden_dir, "geom_bboxes_golden.txt")
	actual_geom = os.path.join(out_dir, "actual_geom_filtered_e2e.txt")
	image_path = find_image_path(stem, image_dir)
	if os.path.exists(golden_geom) and os.path.exists(actual_geom):
		verify_geom_filter(golden_geom, actual_geom, out_dir, image_dir, image_path)

	golden_tmpl = os.path.join(golden_dir, "template_matching_golden.txt")
	actual_tmpl = os.path.join(out_dir, "actual_template_matching_e2e.txt")
	if os.path.exists(golden_tmpl) and os.path.exists(actual_tmpl) and image_path:
		verify_roi_template(golden_tmpl, actual_tmpl, out_dir, image_path, template_dir)


def main():
	parser = argparse.ArgumentParser(description="Generate fresh E2E visuals per frame.")
	parser.add_argument("--project-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
	parser.add_argument("--golden-root", default=None)
	parser.add_argument("--out-root", default=None)
	parser.add_argument("--image-dir", default=None)
	parser.add_argument("--template-dir", default=None)
	parser.add_argument("--frames", default=None, help="Comma-separated list of frame filenames")
	parser.add_argument("--frame-list-file", default=None, help="Text file with one frame filename per line")
	args = parser.parse_args()

	project_root = args.project_root
	golden_root = args.golden_root or os.path.join(project_root, "results", "by_image")
	out_root = args.out_root or os.path.join(project_root, "results", "by_image")
	image_dir = args.image_dir or os.path.join(project_root, "pyton", "pics_bank")
	template_dir = choose_template_dir(project_root, args.template_dir)

	stems = []
	if args.frames:
		for item in args.frames.split(","):
			name = item.strip()
			if name:
				stem = os.path.splitext(os.path.basename(name))[0]
				if not should_skip_stem(stem):
					stems.append(stem)
	if args.frame_list_file:
		with open(args.frame_list_file, "r", encoding="utf-8") as handle:
			for line in handle:
				s = line.strip()
				if not s or s.startswith("#"):
					continue
				stem = os.path.splitext(os.path.basename(s))[0]
				if not should_skip_stem(stem):
					stems.append(stem)
	if not stems:
		stems = [stem for stem in list_e2e_stems(out_root) if not should_skip_stem(stem)]

	if not stems:
		raise SystemExit("No frames found. Provide --frames or generate E2E outputs first.")

	for stem in stems:
		run_for_stem(stem, golden_root, out_root, image_dir, template_dir)


if __name__ == "__main__":
	main()
