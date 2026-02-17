import os
import math
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
import pandas as pd

#################################################################

def save_fedback_to_excel(train_path, text, correct_label):
    row = {
        "text": text,
        "label": correct_label
        }

    new_df = pd.DataFrame([row])

    if os.path.exists(train_path):
        existing = pd.read_excel(train_path, engine="openpyxl")
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_excel(train_path, index=False, engine="openpyxl")


def hitl_ui(pred_label, confidence):
    feedback = {
        "status": 'cancel',
        "correct_label": None
    }

    root = tk.Tk()
    root.withdraw()  # hide window

    # Initialization
    dialog = tk.Toplevel(root)
    dialog.title("Verification necessary")
    dialog.resizable(True, True)
    dialog.grab_set()  # modal

    # Window size
    dialog.update_idletasks()
    width, height = 420, 220
    x = (dialog.winfo_screenwidth() // 2) - (width // 2)
    y = (dialog.winfo_screenheight() // 2) - (height // 2)
    dialog.geometry(f"{width}x{height}+{x}+{y}")

    # Contente
    container = tk.Frame(dialog, padx=16, pady=14)
    container.pack(fill="both", expand=True)

    header = tk.Label(container, text="Please verify E3 element match", font=("TkDefaultFont", 10, "bold"))
    header.pack(anchor="w", pady=(0, 8))

    info = tk.Label(container, text=f"Prediction: {pred_label}    •    Confidence: {confidence:.2%}")
    info.pack(anchor="w", pady=(0, 6))

    # Buttons
    btn_row = tk.Frame(container)
    btn_row.pack(fill="x", pady=(8, 0))

    # Correction frame
    correction_frame = tk.Frame(container)
    correction_label = tk.Label(correction_frame, text="Please write correct E3 name:")
    correction_entry = tk.Entry(correction_frame)
    save_btn = tk.Button(correction_frame, text="Save", width=12)
    cancel_btn2 = tk.Button(correction_frame, text="Cancel", width=12)

    def on_correct():
        feedback["status"] = 'correct'
        feedback["correct_label"] = None
        dialog.destroy()

    def on_wrong():
        correction_frame.pack(fill="x", pady=(12, 0))
        correction_label.pack(anchor="w")
        correction_entry.pack(fill="x", pady=(6, 6))
        correction_entry.focus_set()

        # Buttons for save/cancel
        actions = tk.Frame(correction_frame)
        actions.pack(fill="x")

        def on_save():
            val = correction_entry.get().strip()
            if val == "":
                messagebox.showwarning("Correction is missing", "Please insert correct E3 name.")
                return
            feedback["status"] = 'incorrect'
            feedback["correct_label"] = val
            dialog.destroy()

        def on_cancel2():
            feedback["status"] = 'cancel'
            feedback["correct_label"] = None
            dialog.destroy()

        nonlocal save_btn, cancel_btn2
        save_btn = tk.Button(actions, text="Save", width=12, command=on_save)
        save_btn.pack(side="right", padx=(8, 0))
        cancel_btn2 = tk.Button(actions, text="Cancel", width=12, command=on_cancel2)
        cancel_btn2.pack(side="right")

    correct_btn = tk.Button(btn_row, text="Correkt", width=12, command=on_correct)
    wrong_btn = tk.Button(btn_row, text="Wrong", width=12, command=on_wrong)
    wrong_btn.pack(side="right")
    correct_btn.pack(side="right", padx=(0, 8))

    # ESC clos as cancel
    dialog.bind("<Escape>", lambda e: dialog.destroy())

    # Blocking wait (modal)
    root.wait_window(dialog)
    root.destroy()

    return feedback


def retrain_svm_from_excel(
    excel_path: str,
    *,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str | float = "scale",
    probability: bool = True,
):
    """
    Lädt die in der Excel gesammelten Daten, trennt Features (f*) und label,
    trainiert eine SVM und gibt das trainierte Modell (Pipeline) zurück.
    Erwartet Spalte 'label' und Feature-Spalten f0, f1, ...
    """
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel nicht gefunden: {excel_path}")

    df = pd.read_excel(excel_path, engine="openpyxl")

    # Feature-Spalten automatisch erkennen (f0, f1, f2, ...)
    feature_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    if not feature_cols:
        raise ValueError("Keine Feature-Spalten (f0, f1, ...) in der Excel gefunden.")
    if "label" not in df.columns:
        raise ValueError("Spalte 'label' fehlt in der Excel.")

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].astype(str).to_numpy()

    # Standard-Setup: Skalierung + SVC
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel, C=C, gamma=gamma, probability=probability, class_weight="balanced"),
    )
    model.fit(X, y)
    return model