
import os
import math
import re

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from PIL import Image


####################################################################################

def find_file_with_same_name(root_dir, filename, recursive = True, sort_by ="folder") -> List[Path]:
    matches: List[Path] = []
    if recursive:
        for p in root_dir.rglob(filename):
            if p.is_file():
                matches.append(p)
    else:
        p = root_dir / filename
        if p.is_file():
            matches.append(p)

    if sort_by == "folder":
        matches.sort(key=lambda p: p.parent.name.lower())
    elif sort_by == "path":
        matches.sort(key=lambda p: str(p).lower())
    elif sort_by == "mtime":
        matches.sort(key=lambda p: p.stat().st_mtime)
    return matches


def plot_values_from_files(file_paths, search_string, save_path):
    values = []
    units = []
    labels = []

    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if search_string in line:
                    # extract float number and unit
                    match = re.search(r":\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]+|%)", line)
                    if match:
                        value = float(match.group(1))
                        unit = match.group(2)
                        label = os.path.basename(os.path.dirname(path))

                        values.append(value)
                        units.append(unit)
                        labels.append(label[11:len(label) - 14])
                        break

    # Check units
    unique_units = set(units)
    unit_label = unique_units.pop() if len(unique_units) == 1 else ", ".join(unique_units)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(values, marker="o")
    plt.xticks(range(len(values)), labels, rotation=30, ha="right")
    plt.ylabel(f"{search_string} [{unit_label}]")
    plt.title(f"Vergleich {search_string}")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Summary pics saved under: {save_path}")


########################## Plot gathering ##########################################

def open_and_maybe_resize(img_path: Path, max_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    if max_size:
        im.thumbnail(max_size, Image.Resampling.LANCZOS)
    return im


def show_images_in_one_figure(
    image_paths: List[Path],
    title: str = "Result comparison",
    caption_mode: str = "folder",  # "folder" | "parent_path"
    max_image_size: Optional[Tuple[int, int]] = (1200, 1200),
    tight_layout: bool = True,
    dpi: int = 200,
    save_path: Optional[Path] = None):

    n = len(image_paths)
    if n == 0:
        raise ValueError("Keine Bilder gefunden – prüfe root_dir und filename.")

    # compute grid
    if n <= 0:
        cols = 0
        rows = 0
    else:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

    # adapt size automatically
    # per column ~4 inch width, per line ~3.5 inch height
    fig_w = max(6, 4 * cols)
    fig_h = max(4, 3.5 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=dpi)
    if n == 1:
        axes = [axes]  # zu Liste normalisieren
    else:
        axes = list(axes.flatten())

    fig.suptitle(title, fontsize=12)

    for ax_idx, ax in enumerate(axes):
        if ax_idx < n:
            img_path = image_paths[ax_idx]
            img = open_and_maybe_resize(img_path, max_size=max_image_size)
            ax.imshow(img)
            ax.axis("off")

            if caption_mode == "folder":
                caption = img_path.parent.name
            elif caption_mode == "parent_path":
                caption = str(img_path.parent)
            else:
                caption = img_path.parent.name

            ax.set_title(caption, fontsize=10)
        else:
            # not show empty cells
            ax.axis("off")

    if tight_layout:
        plt.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)
        print(f"Summary pics saved under: {save_path}")


def gather_pictures(keyword, addition, root):
    filenames = ["confusion_matrix_sts.png", "confusion_matrix_svm.png", "cpu_usage.png", "learning_curve.png", "performance.png"]

    for filename in filenames:
        paths = find_file_with_same_name(root_dir=root, filename=filename, recursive=True, sort_by="folder")

        if keyword:
            paths_filtered = []
            for path in paths:
                if keyword in str(path):
                    paths_filtered.append(path)
                else: pass
            paths = paths_filtered

        show_images_in_one_figure(image_paths=paths, title=f"Comparison: {filename}", caption_mode="folder",
                                  max_image_size=(1200, 1200), tight_layout=True, dpi=200,
                                  save_path=root / f"vergleich_allCabs_bilder_{addition}_{filename}")


########################## Cross validation gathering ##############################

def gather_log(keyword, search_string, addition, root):
    filename = "evaluation_log.txt"

    paths = find_file_with_same_name(root_dir=root, filename=filename, recursive=True, sort_by="folder")

    if keyword:
        paths_filtered = []
        for path in paths:
            if keyword in str(path):
                paths_filtered.append(path)
            else:
                pass
        paths = paths_filtered

    plot_values_from_files(paths, search_string, save_path=root / f"vergleich_allCabs_{addition}_{search_string}")

####################################################################################
######################### Single execution #########################################

root = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Evaluation"))
# gather_pictures("_scaled_", root)
# gather_log("_scaled_", "inference RAM", root)
