"""Small initiator window for the collected MC + seg pipeline.

    python lab/pipeline/run_collected.py --gui
"""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from lab.configs.defaults import (
    DEFAULT_INPUT_TIFF_TEMPLATE,
    SENSORS,
    mapping_from_emission,
    microscope_for_path,
    pmt_colors,
)
from lab.pipeline.run_collected import infer_cell_type, run_tree


class _LogWriter:
    def __init__(self, widget: tk.Text):
        self.widget = widget

    def write(self, text):
        if not text:
            return
        self.widget.after(0, self._append, text)

    def _append(self, text):
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)

    def flush(self):
        pass


def launch(args=None):
    root = tk.Tk()
    root.title("suite2p  ·  MC + seg / traces")
    root.geometry("720x560")
    app = Initiator(root, args)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()
    return 0


class Initiator(ttk.Frame):
    def __init__(self, master, args=None):
        super().__init__(master, padding=10)
        self._busy = False
        self._syncing = False

        self.root_var = tk.StringVar(value=(getattr(args, "root", None) or ""))
        self.template_var = tk.StringVar(
            value=getattr(args, "input_template", None) or DEFAULT_INPUT_TIFF_TEMPLATE
        )
        self.sensor_green = tk.StringVar(
            value=getattr(args, "green_sensor", None) or "G-Flamp"
        )
        self.sensor_red = tk.StringVar(
            value=getattr(args, "red_sensor", None) or "jRGECO"
        )
        self.cell_green = tk.StringVar(
            value=getattr(args, "green_cell", None) or "astrocyte"
        )
        self.cell_red = tk.StringVar(
            value=getattr(args, "red_cell", None) or "neuron"
        )
        self.do_mc = tk.BooleanVar(value=True)
        self.do_seg = tk.BooleanVar(value=True)
        self.overwrite = tk.BooleanVar(value=False)
        self.save_stack = tk.BooleanVar(value=False)
        self.min_nframes = tk.IntVar(value=int(getattr(args, "min_nframes", None) or 500))
        self.microscope_var = tk.StringVar(value="(pick a root to read Experiment.xml)")

        self._build()
        self.sensor_green.trace_add("write", lambda *_: self._sensor_to_cell("green"))
        self.sensor_red.trace_add("write", lambda *_: self._sensor_to_cell("red"))
        if self.root_var.get():
            self._apply_xml_defaults(Path(self.root_var.get()))

    def _build(self):
        row = 0
        ttk.Label(self, text="Root directory (walk for sessions)").grid(
            row=row, column=0, sticky="w"
        )
        row += 1
        ttk.Entry(self, textvariable=self.root_var, width=70).grid(
            row=row, column=0, columnspan=3, sticky="ew"
        )
        ttk.Button(self, text="Browse…", command=self._browse).grid(row=row, column=3, padx=4)
        row += 1
        ttk.Label(self, textvariable=self.microscope_var, foreground="#444").grid(
            row=row, column=0, columnspan=4, sticky="w", pady=(2, 8)
        )
        row += 1

        ttk.Label(self, text="Green").grid(row=row, column=0, sticky="w")
        ttk.Combobox(
            self, textvariable=self.sensor_green, values=SENSORS, width=12, state="readonly"
        ).grid(row=row, column=1, sticky="w")
        ttk.Combobox(
            self, textvariable=self.cell_green, values=("neuron", "astrocyte"), width=12,
            state="readonly",
        ).grid(row=row, column=2, sticky="w")
        ttk.Label(self, text="sensor / cell type").grid(row=row, column=3, sticky="w")
        row += 1
        ttk.Label(self, text="Red").grid(row=row, column=0, sticky="w")
        ttk.Combobox(
            self, textvariable=self.sensor_red, values=SENSORS, width=12, state="readonly"
        ).grid(row=row, column=1, sticky="w")
        ttk.Combobox(
            self, textvariable=self.cell_red, values=("neuron", "astrocyte"), width=12,
            state="readonly",
        ).grid(row=row, column=2, sticky="w")
        row += 1

        ttk.Label(self, text="MC input TIFF").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(self, textvariable=self.template_var, width=40).grid(
            row=row, column=1, columnspan=3, sticky="ew", pady=(8, 0)
        )
        row += 1
        ttk.Label(
            self,
            text=(
                "Green/red stay with the prep. XML maps them onto ChanA/B "
                "(Shinano A=red, Musashi A=green). Writes suite2p_temp / suite2p_anat "
                "into DATA/ChanA|B; does not overwrite the v22 TIFFs."
            ),
            foreground="#444",
        ).grid(row=row, column=0, columnspan=4, sticky="w")
        row += 1

        opts = ttk.Frame(self)
        opts.grid(row=row, column=0, columnspan=4, sticky="w", pady=8)
        ttk.Checkbutton(opts, text="Motion correction", variable=self.do_mc).pack(side=tk.LEFT)
        ttk.Checkbutton(opts, text="Seg + traces", variable=self.do_seg).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(opts, text="Overwrite", variable=self.overwrite).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(opts, text="Write registered TIFFs", variable=self.save_stack).pack(
            side=tk.LEFT, padx=8
        )
        ttk.Label(opts, text="min frames").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Spinbox(opts, from_=0, to=20000, increment=100, width=6,
                    textvariable=self.min_nframes).pack(side=tk.LEFT)
        row += 1

        btns = ttk.Frame(self)
        btns.grid(row=row, column=0, columnspan=4, sticky="w", pady=4)
        ttk.Button(btns, text="Inventory", command=lambda: self._go(inventory=True)).pack(
            side=tk.LEFT
        )
        ttk.Button(btns, text="Run", command=lambda: self._go(inventory=False)).pack(
            side=tk.LEFT, padx=8
        )
        row += 1

        self.log = tk.Text(self, height=16, wrap=tk.WORD)
        self.log.grid(row=row, column=0, columnspan=4, sticky="nsew", pady=(8, 0))
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(row, weight=1)

    def _browse(self):
        path = filedialog.askdirectory(title="Select experiment root")
        if path:
            self.root_var.set(path)
            self._apply_xml_defaults(Path(path))

    def _apply_xml_defaults(self, start: Path):
        info = microscope_for_path(start)
        if info is None:
            xml = next(Path(start).rglob("Experiment.xml"), None)
            info = microscope_for_path(xml.parent) if xml else None
        if info is None:
            self.microscope_var.set("No Experiment.xml found yet (will search under root)")
            return
        name = info.get("name") or "unknown"
        computer = info.get("computer") or ""
        colors = pmt_colors(computer)
        self.microscope_var.set(
            f"XML: {name}  ({computer})  — Chan{colors['green']}=green, "
            f"Chan{colors['red']}=red  (fs / µm/px at run time)"
        )

    def _sensor_to_cell(self, color: str):
        if self._syncing:
            return
        sensor = self.sensor_green.get() if color == "green" else self.sensor_red.get()
        inferred = infer_cell_type(sensor, "astrocyte" if color == "green" else "neuron")
        if color == "green":
            self.cell_green.set(inferred)
        else:
            self.cell_red.set(inferred)

    def _go(self, inventory: bool):
        if self._busy:
            return
        root = self.root_var.get().strip()
        if not root or not Path(root).exists():
            messagebox.showerror("Missing root", "Pick an existing root directory.")
            return
        try:
            mapping_from_emission(self.cell_green.get(), self.cell_red.get())
        except ValueError as exc:
            messagebox.showerror("Cell types", str(exc))
            return
        self._busy = True
        self.log.delete("1.0", tk.END)
        writer = _LogWriter(self.log)

        def work():
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = writer
            try:
                run_tree(
                    Path(root),
                    template=self.template_var.get().strip() or DEFAULT_INPUT_TIFF_TEMPLATE,
                    green_cell=self.cell_green.get(),
                    red_cell=self.cell_red.get(),
                    green_sensor=self.sensor_green.get(),
                    red_sensor=self.sensor_red.get(),
                    do_mc=self.do_mc.get(),
                    do_seg=self.do_seg.get(),
                    overwrite=self.overwrite.get(),
                    inventory=inventory,
                    write_registered_tif=self.save_stack.get(),
                    min_nframes=int(self.min_nframes.get() or 0),
                )
            except Exception as exc:
                print(f"\nFAILED: {exc}")
            finally:
                sys.stdout, sys.stderr = old_out, old_err
                self._busy = False

        threading.Thread(target=work, daemon=True).start()
