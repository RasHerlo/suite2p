"""Simple GUI: motion-correct one TIFF stack with stock Suite2p registration."""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import mc_core


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Suite2p Motion Correction")
        self.geometry("640x560")
        self.q: queue.Queue = queue.Queue()
        self.worker = None

        pad = {"padx": 8, "pady": 4}
        f = ttk.Frame(self)
        f.pack(fill="x", **pad)
        self.in_var, self.out_var = tk.StringVar(), tk.StringVar()
        for r, (label, var, cmd) in enumerate(
                [("TIFF stack", self.in_var, self.pick_in),
                 ("Output folder", self.out_var, self.pick_out)]):
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", **pad)
            ttk.Entry(f, textvariable=var, width=60).grid(row=r, column=1, **pad)
            ttk.Button(f, text="Browse...", command=cmd).grid(row=r, column=2, **pad)

        o = ttk.LabelFrame(self, text="Options (Suite2p registration)")
        o.pack(fill="x", **pad)
        d = mc_core.DEFAULT_PARAMS
        self.nonrigid = tk.BooleanVar(value=d["nonrigid"])
        self.bidi = tk.BooleanVar(value=d["do_bidiphase"])
        self.onep = tk.BooleanVar(value=d["1Preg"])
        ttk.Checkbutton(o, text="Non-rigid", variable=self.nonrigid).grid(row=0, column=0, sticky="w", **pad)
        ttk.Checkbutton(o, text="Bidiphase correction", variable=self.bidi).grid(row=0, column=1, sticky="w", **pad)
        ttk.Checkbutton(o, text="1P mode", variable=self.onep).grid(row=0, column=2, sticky="w", **pad)
        self.nums = {}
        for i, (key, label) in enumerate([("block_size", "Block size"),
                                          ("maxregshift", "Max shift (fraction)"),
                                          ("smooth_sigma", "Smooth sigma"),
                                          ("nimg_init", "Frames for reference"),
                                          ("batch_size", "Batch size")]):
            r, c = divmod(i, 3)
            ttk.Label(o, text=label).grid(row=r + 1, column=c * 2, sticky="w", **pad)
            v = tk.StringVar(value=str(d[key]))
            ttk.Entry(o, textvariable=v, width=8).grid(row=r + 1, column=c * 2 + 1, sticky="w", **pad)
            self.nums[key] = v

        b = ttk.Frame(self)
        b.pack(fill="x", **pad)
        self.run_btn = ttk.Button(b, text="Run motion correction", command=self.run)
        self.run_btn.pack(side="left")
        self.bar = ttk.Progressbar(b, mode="indeterminate", length=250)
        self.bar.pack(side="left", padx=12)

        self.log = tk.Text(self, height=18, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True, **pad)
        self.after(100, self.poll)
        self.write("Pick a TIFF stack (either channel) and press Run.")

    def pick_in(self):
        p = filedialog.askopenfilename(filetypes=[("TIFF", "*.tif *.tiff"), ("All", "*.*")])
        if p:
            self.in_var.set(p)
            if not self.out_var.get():
                self.out_var.set(str(Path(p).parent / "motion_corrected"))

    def pick_out(self):
        p = filedialog.askdirectory()
        if p:
            self.out_var.set(p)

    def write(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def poll(self):
        try:
            while True:
                kind, msg = self.q.get_nowait()
                if kind == "log":
                    self.write(msg)
                else:
                    self.bar.stop()
                    self.run_btn.configure(state="normal")
                    if kind == "error":
                        self.write("ERROR: " + msg)
                        messagebox.showerror("Motion correction failed", msg)
                    else:
                        messagebox.showinfo("Done", msg)
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def run(self):
        tif = self.in_var.get().strip()
        if not tif or not Path(tif).is_file():
            messagebox.showwarning("Input", "Choose a TIFF stack first.")
            return
        try:
            params = {
                "nonrigid": self.nonrigid.get(),
                "do_bidiphase": self.bidi.get(),
                "1Preg": self.onep.get(),
                "block_size": int(self.nums["block_size"].get()),
                "maxregshift": float(self.nums["maxregshift"].get()),
                "smooth_sigma": float(self.nums["smooth_sigma"].get()),
                "nimg_init": int(self.nums["nimg_init"].get()),
                "batch_size": int(self.nums["batch_size"].get()),
            }
        except ValueError:
            messagebox.showwarning("Options", "Check the numeric options.")
            return
        out = self.out_var.get().strip() or None
        self.run_btn.configure(state="disabled")
        self.bar.start(15)

        def work():
            try:
                res = mc_core.motion_correct(
                    tif, out, params, log=lambda m: self.q.put(("log", m)))
                self.q.put(("done", f"Saved:\n{res['registered']}\n{res['shifts']}"))
            except Exception as e:  # noqa: BLE001
                self.q.put(("error", str(e)))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()


def _cli(tif, out):
    """Headless run: app.exe --cli in.tif out_dir  (log written to out_dir/cli_log.txt)."""
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / "cli_log.txt", "w") as lf:
        def log(m):
            lf.write(m + "\n")
            lf.flush()
        try:
            mc_core.motion_correct(tif, out, log=log)
        except Exception as e:  # noqa: BLE001
            import traceback
            log("ERROR: " + traceback.format_exc())


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "--cli":
        _cli(sys.argv[2], sys.argv[3])
    else:
        App().mainloop()
