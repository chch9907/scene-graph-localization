from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np



def _load_errors(log_path: Path) -> Dict[str, np.ndarray]:
	with log_path.open("r", encoding="utf-8") as fp:
		data = json.load(fp)

	try:
		xy_errors = np.asarray(data["xy_errors"], dtype=float)
		yaw_errors = np.asarray(data["yaw_errors"], dtype=float)
	except KeyError as exc:
		raise KeyError(f"Missing key {exc} in {log_path}") from exc

	return {"xy": xy_errors, "yaw": yaw_errors}


def _normalize_label(raw_label: str) -> str:
	base, _, suffix = raw_label.rpartition("_")
	if base and suffix.isdigit():
		return base
	return raw_label


def _infer_label(log_path: Path) -> str:
	stem = log_path.stem
	prefix = "localization_log"
	if stem == prefix:
		return "baseline"
	if stem.startswith(f"{prefix}_"):
		return _normalize_label(stem[len(prefix) + 1 :])
	return _normalize_label(stem)


def _group_log_files_by_label(log_files: List[Path]) -> Dict[str, List[Path]]:
	grouped: Dict[str, List[Path]] = {}
	for path in log_files:
		label = _infer_label(path)
		grouped.setdefault(label, []).append(path)
	for paths in grouped.values():
		paths.sort()
	return grouped


def _average_series(series_list: List[np.ndarray]) -> np.ndarray:
	if not series_list:
		raise ValueError("No series provided for averaging.")
	if len(series_list) == 1:
		return series_list[0]
	max_len = max(arr.shape[0] for arr in series_list)
	stacked = np.full((len(series_list), max_len), np.nan, dtype=float)
	for idx, arr in enumerate(series_list):
		stacked[idx, : arr.shape[0]] = arr
	return np.nanmean(stacked, axis=0)


def _average_errors_across_runs(log_paths: List[Path]) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
	xy_series: List[np.ndarray] = []
	yaw_series: List[np.ndarray] = []
	xy_means: List[float] = []
	yaw_means: List[float] = []
	for path in log_paths:
		errors = _load_errors(path)
		xy_series.append(errors["xy"])
		yaw_series.append(errors["yaw"])
		xy_means.append(float(np.nanmean(errors["xy"])) if errors["xy"].size > 0 else float("nan"))
		yaw_means.append(float(np.nanmean(errors["yaw"])) if errors["yaw"].size > 0 else float("nan"))
	averaged = {"xy": _average_series(xy_series), "yaw": _average_series(yaw_series)}
	stats = {
		"xy_means": np.asarray(xy_means, dtype=float),
		"yaw_means": np.asarray(yaw_means, dtype=float),
	}
	return averaged, stats

caption_dict = {
	'baseline': 'Baseline',
	'tfidf_sem': 'Semantic-only (' + r'$\alpha=1$)',
	'tfidf_cent': 'Centrality-only (' + r'$\alpha=0$)',
	'tfidf_0.3': 'Combined (' + r'$\alpha=0.3$)',
	'tfidf_0.5': 'Combined (' + r'$\alpha=0.5$)',
	'tfidf_0.7': 'Combined (' + r'$\alpha=0.7$)',
}

def plot_errors(
	label_errors: Dict[str, Dict[str, np.ndarray]],
	labels: List[str],
	output: Path,
	title: str,
	show: bool,
) -> None:
	fig, (ax_xy, ax_yaw) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

	for label in labels:
		errors = label_errors.get(label)
		if errors is None:
			continue
		caption = caption_dict[label] if label in caption_dict else label
		print(label, caption)
		steps_xy = np.arange(errors["xy"].shape[0])
		steps_yaw = np.arange(errors["yaw"].shape[0])
		ax_xy.plot(steps_xy, errors["xy"], label=caption)
		ax_yaw.plot(steps_yaw, errors["yaw"], label=caption)

	ax_xy.set_ylabel("XY Error (m)")
	ax_xy.set_title(title)
	ax_xy.grid(True, linestyle="--", alpha=0.4)
	ax_xy.legend()

	ax_yaw.set_ylabel("Yaw Error (deg)")
	ax_yaw.set_xlabel("Step")
	ax_yaw.grid(True, linestyle="--", alpha=0.4)

	fig.tight_layout()
	fig.savefig(output, dpi=200)
	if show:
		plt.show()


def report_metrics(
	label_errors: Dict[str, Dict[str, np.ndarray]],
	labels: List[str],
	label_counts: Dict[str, int],
	label_stats: Dict[str, Dict[str, np.ndarray]],
) -> None:
	print("\nAverage localization errors:")
	print("label\txy_mean(m)\tyaw_mean(deg)\tstd_xy\tstd_yaw")
	for label in labels:
		errors = label_errors.get(label)
		if errors is None:
			continue
		xy_mean = float(np.nanmean(errors["xy"])) if errors["xy"].size > 0 else float("nan")
		yaw_mean = float(np.nanmean(errors["yaw"])) if errors["yaw"].size > 0 else float("nan")
		count = label_counts.get(label, 1)
		stats = label_stats.get(label, {})
		if count > 1 and stats:
			x_vals = stats.get("xy_means")
			y_vals = stats.get("yaw_means")
			xy_std_val = float(np.nanstd(x_vals)) if x_vals is not None and x_vals.size else float("nan")
			yaw_std_val = float(np.nanstd(y_vals)) if y_vals is not None and y_vals.size else float("nan")
			xy_std = f"{xy_std_val:.3f}" if not np.isnan(xy_std_val) else "nan"
			yaw_std = f"{yaw_std_val:.3f}" if not np.isnan(yaw_std_val) else "nan"
		else:
			xy_std = "-"
			yaw_std = "-"
		print(f"{label}\t{xy_mean:.3f}\t{yaw_mean:.3f}\t{xy_std}\t{yaw_std}")


def discover_log_files(log_dir: Path) -> List[Path]:
	if not log_dir.exists():
		raise FileNotFoundError(f"Log directory {log_dir} does not exist")
	pattern = "localization_log*.json"
	candidates = sorted(log_dir.glob(pattern))
	if not candidates:
		raise FileNotFoundError(f"No log files matching {pattern} found in {log_dir}")
	return candidates


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot localization errors for multiple methods.")
	parser.add_argument("--log-dir", required=True, help="Directory containing localization log JSON files.")
	parser.add_argument("--labels", nargs="+", help="Legend labels for each log file.")
	parser.add_argument(
		"--output",
		default="localization_errors_compare.png",
		help="Output path for the generated figure.",
	)
	parser.add_argument("--title", default="Localization Errors vs Steps", help="Figure title.")
	parser.add_argument("--show", action="store_true", help="Display the plot interactively.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	log_dir = Path(args.log_dir).expanduser().resolve()
	log_files = discover_log_files(log_dir)
	grouped_logs = _group_log_files_by_label(log_files)
	print(f"Discovered log files for labels: {grouped_logs}")
	requested_labels = args.labels or list(grouped_logs.keys())
	labels: List[str] = []
	label_errors: Dict[str, Dict[str, np.ndarray]] = {}
	label_counts: Dict[str, int] = {}
	label_stats: Dict[str, Dict[str, np.ndarray]] = {}
	for label in requested_labels:
		if label in label_errors:
			continue
		paths = grouped_logs.get(label)
		if not paths:
			print(f"Expected log files for label '{label}' not found in {log_dir}")
			continue
		averaged_errors, stats = _average_errors_across_runs(paths)
		label_errors[label] = averaged_errors
		label_counts[label] = len(paths)
		label_stats[label] = stats
		labels.append(label)

	if not labels:
		raise ValueError("No matching log files found for the provided labels.")

	output = log_dir / Path(args.output)
	output.parent.mkdir(parents=True, exist_ok=True)
	plot_errors(label_errors, labels, output, args.title, args.show)
	report_metrics(label_errors, labels, label_counts, label_stats)


if __name__ == "__main__":
	main()
