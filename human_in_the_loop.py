import os
import tkinter as tk
from tkinter import messagebox
import pandas as pd

from setup_vector_database import *

#################################################################

def save_fedback_to_excel(train_path, text, cab, correct_label):

    row = {
        "Text": text,
        "Label": correct_label,
        "Cab": cab,
        "New": "new"
        }

    new_df = pd.DataFrame([row])

    if os.path.exists(train_path):
        existing = pd.read_excel(train_path, engine="openpyxl")
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_excel(train_path, index=False, engine="openpyxl")


def hitl_ui(pred_label, confidence, persistent_dir, train_path):

    df = pd.read_excel(train_path, engine="openpyxl")
    new_cnt = df["New"].count()

    classes = get_all_classes(persistent_dir)
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

    # Content
    container = tk.Frame(dialog, padx=16, pady=14)
    container.pack(fill="both", expand=True)

    header = tk.Label(container, text="Please verify the following E3 element match", font=("TkDefaultFont", 10, "bold"))
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
    retrain_frame = tk.Frame(container)
    retrain_label = tk.Label(retrain_frame, text=f"Currently {new_cnt} new entries in training database")
    save_btn1 = tk.Button(correction_frame, text="Save", width=12)
    cancel_btn1 = tk.Button(correction_frame, text="Cancel", width=12)
    save_btn2 = tk.Button(retrain_frame, text="Retrain", width=12)
    cancel_btn2 = tk.Button(retrain_frame, text="Cancel", width=12)

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
            elif val not in classes:
                messagebox.showwarning("E3 name is not matching any database entry", "Please insert correct E3 name.")
                return

            feedback["status"] = 'incorrect'
            feedback["correct_label"] = val
            dialog.destroy()

        def on_cancel2():
            feedback["status"] = 'cancel'
            feedback["correct_label"] = None
            dialog.destroy()

        nonlocal save_btn1, cancel_btn1
        save_btn1 = tk.Button(actions, text="Save", width=12, command=on_save)
        save_btn1.pack(side="right", padx=(8, 0))
        cancel_btn1 = tk.Button(actions, text="Cancel", width=12, command=on_cancel2)
        cancel_btn1.pack(side="right")

    def on_retrain():
        retrain_frame.pack(fill="x", pady=(12, 0))
        retrain_label.pack(anchor="w")
        feedback["status"] = 'retrain'

        # Buttons for retrain/cancel
        actions = tk.Frame(retrain_frame)
        actions.pack(fill="x")

        def on_retrain2():
            feedback["status"] = 'retrain'
            feedback["correct_label"] = None
            dialog.destroy()

        def on_cancel3():
            feedback["status"] = 'cancel'
            feedback["correct_label"] = None
            dialog.destroy()

        nonlocal save_btn2, cancel_btn2
        save_btn2 = tk.Button(actions, text="Retrain", width=12, command=on_retrain2)
        save_btn2.pack(side="right", padx=(8, 0))
        cancel_btn2 = tk.Button(actions, text="Cancel", width=12, command=on_cancel3)
        cancel_btn2.pack(side="right")

    correct_btn = tk.Button(btn_row, text="Correkt", width=12, command=on_correct)
    wrong_btn = tk.Button(btn_row, text="Wrong", width=12, command=on_wrong)
    retrain_btn = tk.Button(btn_row, text="Retrain", width=12, command=on_retrain)
    wrong_btn.pack(side="left", padx=(0, 6))
    correct_btn.pack(side="left")
    retrain_btn.pack(side="right")

    # ESC close as cancel
    dialog.bind("<Escape>", lambda e: dialog.destroy())

    # Blocking wait (modal)
    root.wait_window(dialog)
    root.destroy()

    return feedback