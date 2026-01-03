
import os
import math
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from PIL import Image

############################## Globals #############################################



####################################################################################
def find_images_with_same_name(root_dir, filename, recursive = True, sort_by = "folder") -> List[Path]:

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


def compute_grid(n: int) -> Tuple[int, int]:

    if n <= 0:
        return (0, 0)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


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
    save_path: Optional[Path] = None,
):

    n = len(image_paths)
    if n == 0:
        raise ValueError("Keine Bilder gefunden – prüfe root_dir und filename.")

    rows, cols = compute_grid(n)

    # Abbildung erstellen – Größe dynamisch anpassen
    # Faustregel: pro Spalte ~4 inch Breite, pro Zeile ~3.5 inch Höhe
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
                # relativen Pfad zur root anzeigen (falls sinnvoll)
                caption = str(img_path.parent)
            else:
                caption = img_path.parent.name  # Fallback

            ax.set_title(caption, fontsize=10)
        else:
            # leere Zellen ausblenden
            ax.axis("off")

    if tight_layout:
        plt.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)
        print(f"Gesamtabbild gespeichert unter: {save_path}")

    plt.show()


if __name__ == "__main__":
    root = str(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Evaluation"))
    filenames = ["confusion_matrix_sts.png", "confusion_matrix_svm.png", "cpu_usage.png", "learning_curve.png", "performance.png"]

    for filename in filenames:
        paths = find_images_with_same_name(
            root_dir=root,
            filename=filename,
            recursive=True,
            sort_by="folder",
        )

        show_images_in_one_figure(
            image_paths=paths,
            title=f"Comparison: {filename}",
            caption_mode="folder",
            max_image_size=(1200, 1200),
            tight_layout=True,
            dpi=200,
            save_path=root / "vergleich_bilder.png",
        )
