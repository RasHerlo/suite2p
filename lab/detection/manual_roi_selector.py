#!/usr/bin/env python3
"""
manual_roi_selector.py
======================
Standalone GUI for reviewing and manually adjusting ROI selection
from a suite2p plane0 output folder.

Usage
-----
    python lab/detection/manual_roi_selector.py
    python lab/detection/manual_roi_selector.py  /path/to/plane0

Keys
----
    ←  /  →     previous / next ROI
    Space        toggle selected state of current ROI

Dependencies
------------
    numpy, matplotlib, pyqtgraph, qtpy (PyQt5 or PySide2 backend)
"""

import sys
import colorsys
import numpy as np
from pathlib import Path
from matplotlib.path import Path as MplPath

from qtpy import QtGui, QtCore
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QCheckBox,
    QSpinBox, QFileDialog, QMessageBox,
    QSlider, QFrame, QSizePolicy,
)
import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder='row-major')

# ─── Style ────────────────────────────────────────────────────────────────────

_DARK = """
QMainWindow, QWidget { background: #1c1c1c; color: #ddd; }
QPushButton {
    background: #383838; color: #ddd;
    border: 1px solid #505050; border-radius: 3px;
    padding: 3px 10px; min-width: 60px;
}
QPushButton:hover   { background: #484848; }
QPushButton:checked, QPushButton:pressed { background: #5c3d8c; border-color: #7a5ab0; }
QPushButton:disabled { background: #272727; color: #484848; border-color: #333; }
QComboBox {
    background: #383838; color: #ddd;
    border: 1px solid #505050; border-radius: 3px; padding: 2px 6px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background: #383838; color: #ddd;
    selection-background-color: #5c3d8c;
}
QSpinBox {
    background: #383838; color: #ddd;
    border: 1px solid #505050; border-radius: 3px; padding: 2px;
}
QCheckBox { color: #ddd; spacing: 6px; }
QLabel    { color: #ddd; }
QSlider::groove:horizontal { height: 4px; background: #484848; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #7a5ab0; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}
"""


# ─── Custom ViewBox: freehand lasso + pixel-click ─────────────────────────────

class LassoViewBox(pg.ViewBox):
    """
    pg.ViewBox augmented with:
      • Freehand lasso drawing (hold left button while in draw mode)
      • sigLassoComplete  — emits np.ndarray (N, 2) of (col, row) points
      • sigPixelClicked   — emits (col, row) on single click outside draw mode
    """

    sigLassoComplete = QtCore.Signal(object)    # np.ndarray (N,2) col,row
    sigPixelClicked  = QtCore.Signal(int, int)  # col, row

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._draw_mode = False
        self._drawing   = False
        self._pts       = []
        self._preview   = None          # PlotDataItem shown while drawing

    # ── Public API ────────────────────────────────────────────────────────────
    def set_draw_mode(self, on):
        self._draw_mode = on
        self.setCursor(QtCore.Qt.CrossCursor if on else QtCore.Qt.ArrowCursor)
        if not on:
            self._clear_preview()

    # ── Mouse events ──────────────────────────────────────────────────────────
    def mousePressEvent(self, ev):
        if self._draw_mode and ev.button() == QtCore.Qt.LeftButton:
            self._drawing = True
            self._pts.clear()
            p = self.mapSceneToView(ev.scenePos())
            self._pts.append((p.x(), p.y()))
            ev.accept()
            return
        if ev.button() == QtCore.Qt.LeftButton:
            p = self.mapSceneToView(ev.scenePos())
            self.sigPixelClicked.emit(int(round(p.x())), int(round(p.y())))
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._draw_mode and self._drawing:
            p = self.mapSceneToView(ev.scenePos())
            self._pts.append((p.x(), p.y()))
            self._update_preview()
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._draw_mode and self._drawing and ev.button() == QtCore.Qt.LeftButton:
            self._drawing = False
            self._clear_preview()
            if len(self._pts) > 5:
                self.sigLassoComplete.emit(np.array(self._pts, dtype=float))
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    # ── Preview ───────────────────────────────────────────────────────────────
    def _update_preview(self):
        self._clear_preview()
        if len(self._pts) < 2:
            return
        a = np.asarray(self._pts)
        self._preview = pg.PlotDataItem(a[:, 0], a[:, 1],
                                        pen=pg.mkPen('y', width=1.5))
        self.addItem(self._preview)

    def _clear_preview(self):
        if self._preview is not None:
            self.removeItem(self._preview)
            self._preview = None


# ─── Main window ──────────────────────────────────────────────────────────────

class ManualROISelector(QMainWindow):

    _BG_KEYS  = ['meanImg', 'meanImgE', 'Vcorr', 'max_proj']
    _BG_NAMES = ['Mean image', 'Mean image (enhanced)',
                 'Correlation map', 'Max projection']

    def __init__(self, plane_dir=None):
        super().__init__()
        self.setWindowTitle('Manual ROI Selector')
        self.setGeometry(40, 40, 1640, 980)
        self.setStyleSheet(_DARK)

        # ── Data ──────────────────────────────────────────────────────────────
        self.plane_dir   = None
        self.stat        = None
        self.F           = None
        self.Fneu        = None
        self.iscell      = np.array([], dtype=bool)   # (N,) bool
        self.ops         = None
        self.n_rois      = 0
        self.Ly          = 0
        self.Lx          = 0
        self.current_roi = 0

        self._colors     = None     # (N, 3) uint8 – one random colour per ROI
        self._bnd_y      = []       # list of int32 arrays – boundary row indices
        self._bnd_x      = []       # list of int32 arrays – boundary col indices
        self._roi_map    = None     # (Ly, Lx) int32  →  ROI index, or -1

        self._bg_raw     = {}       # key → (Ly, Lx) float64
        self._bg_key     = 'meanImg'

        self._lasso_polys = []      # list of (K, 2) float arrays – drawn polygons
        self._lasso_mask  = None    # (Ly, Lx) bool  – True = inside selected area

        self._build_ui()

        if plane_dir is not None:
            self._load(Path(plane_dir))

    # ─── Build UI ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(4)

        vbox.addLayout(self._make_toolbar())
        vbox.addLayout(self._make_image_row(), stretch=5)
        vbox.addLayout(self._make_roi_bar())
        vbox.addLayout(self._make_trace_row(), stretch=2)

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _make_toolbar(self):
        bar = QHBoxLayout()
        bar.setSpacing(6)

        self.btn_load = QPushButton('Load plane0')
        self.btn_load.clicked.connect(self._on_load)
        bar.addWidget(self.btn_load)

        self.btn_save = QPushButton('Save iscell.npy')
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._on_save)
        bar.addWidget(self.btn_save)

        bar.addWidget(_vsep())

        self.btn_lasso = QPushButton('Select Area')
        self.btn_lasso.setCheckable(True)
        self.btn_lasso.setEnabled(False)
        self.btn_lasso.toggled.connect(self._on_lasso_toggle)
        bar.addWidget(self.btn_lasso)

        self.btn_clear_lasso = QPushButton('Clear Area')
        self.btn_clear_lasso.setEnabled(False)
        self.btn_clear_lasso.clicked.connect(self._on_clear_lasso)
        bar.addWidget(self.btn_clear_lasso)

        bar.addWidget(_vsep())

        self.btn_all  = QPushButton('Select All')
        self.btn_none = QPushButton('Deselect All')
        for btn, val in ((self.btn_all, True), (self.btn_none, False)):
            btn.setEnabled(False)
            btn.clicked.connect(lambda _checked, v=val: self._bulk(v))
            bar.addWidget(btn)

        bar.addWidget(_vsep())

        bar.addWidget(QLabel('Background:'))
        self.combo_bg = QComboBox()
        self.combo_bg.addItems(self._BG_NAMES)
        self.combo_bg.setFixedWidth(220)
        self.combo_bg.setEnabled(False)
        self.combo_bg.currentIndexChanged.connect(self._on_bg_changed)
        bar.addWidget(self.combo_bg)

        bar.addWidget(_vsep())

        bar.addWidget(QLabel('Sat min:'))
        self.sld_min = _slider(0, 990, 10)
        self.sld_min.setEnabled(False)
        self.sld_min.valueChanged.connect(self._on_sat)
        bar.addWidget(self.sld_min)

        bar.addWidget(QLabel('max:'))
        self.sld_max = _slider(10, 1000, 990)
        self.sld_max.setEnabled(False)
        self.sld_max.valueChanged.connect(self._on_sat)
        bar.addWidget(self.sld_max)

        bar.addStretch()

        self.lbl_status = QLabel('No data loaded')
        self.lbl_status.setStyleSheet('color: #888; font-style: italic;')
        bar.addWidget(self.lbl_status)

        return bar

    # ── Image panels ──────────────────────────────────────────────────────────

    def _make_image_row(self):
        row = QHBoxLayout()
        row.setSpacing(6)

        for lbl_attr, gw_attr, title in [
            ('lbl_left',  'gw_left',  'Selected (0)'),
            ('lbl_right', 'gw_right', 'Not selected (0)'),
        ]:
            col = QVBoxLayout()
            col.setSpacing(2)

            lbl = QLabel(title)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            setattr(self, lbl_attr, lbl)
            col.addWidget(lbl)

            gw = pg.GraphicsLayoutWidget()
            gw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            setattr(self, gw_attr, gw)
            col.addWidget(gw)

            row.addLayout(col)

        self._init_viewboxes()
        return row

    def _init_viewboxes(self):
        panels = [
            ('gw_left',  'vb_left',  'img_bg_l', 'img_roi_l', 'img_lasso_l'),
            ('gw_right', 'vb_right', 'img_bg_r', 'img_roi_r', 'img_lasso_r'),
        ]
        for gw_attr, vb_attr, bg_attr, roi_attr, lasso_attr in panels:
            gw = getattr(self, gw_attr)

            vb = LassoViewBox(invertY=True, lockAspect=True)
            setattr(self, vb_attr, vb)

            # addPlot passes viewBox to PlotItem, which uses our custom vb
            pi = gw.addPlot(row=0, col=0, viewBox=vb)
            pi.hideAxis('left')
            pi.hideAxis('bottom')
            pi.setContentsMargins(0, 0, 0, 0)

            for attr, z in [(bg_attr, 0), (roi_attr, 1), (lasso_attr, 2)]:
                img = pg.ImageItem()
                img.setZValue(z)
                if z > 0:
                    img.setCompositionMode(
                        QtGui.QPainter.CompositionMode_SourceOver)
                setattr(self, attr, img)
                vb.addItem(img)

            vb.sigLassoComplete.connect(self._on_lasso_complete)
            vb.sigPixelClicked.connect(self._on_img_click)

    # ── ROI selector bar ──────────────────────────────────────────────────────

    def _make_roi_bar(self):
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addStretch()

        row.addWidget(QLabel('ROI #:'))

        self.spin = QSpinBox()
        self.spin.setRange(0, 0)
        self.spin.setFixedWidth(75)
        self.spin.setEnabled(False)
        self.spin.valueChanged.connect(self._on_spin)
        row.addWidget(self.spin)

        self.lbl_of = QLabel('/ 0')
        row.addWidget(self.lbl_of)

        self.chk = QCheckBox('Selected')
        self.chk.setEnabled(False)
        self.chk.stateChanged.connect(self._on_toggle)
        row.addWidget(self.chk)

        row.addStretch()
        return row

    # ── Trace panels ──────────────────────────────────────────────────────────

    def _make_trace_row(self):
        row = QHBoxLayout()
        row.setSpacing(6)

        self.plt_f = pg.PlotWidget()
        self.plt_f.showGrid(y=True, alpha=0.25)
        self.plt_f.addLegend(offset=(5, 5))
        row.addWidget(self.plt_f)

        self.plt_d = pg.PlotWidget()
        self.plt_d.showGrid(y=True, alpha=0.25)
        row.addWidget(self.plt_d)

        return row

    # ─── Load / Save ──────────────────────────────────────────────────────────

    def _on_load(self):
        path = QFileDialog.getExistingDirectory(
            self, 'Select plane0 folder', '', QFileDialog.ShowDirsOnly)
        if path:
            self._load(Path(path))

    def _load(self, d):
        d = Path(d)
        needed  = ['stat.npy', 'F.npy', 'Fneu.npy', 'iscell.npy', 'ops.npy']
        missing = [f for f in needed if not (d / f).exists()]
        if missing:
            QMessageBox.critical(
                self, 'Missing files',
                'Cannot find in {}:\n  {}'.format(d, '\n  '.join(missing)))
            return

        self._status('Loading…')
        QApplication.processEvents()

        try:
            self.plane_dir = d
            self.stat  = np.load(d / 'stat.npy',   allow_pickle=True)
            self.F     = np.load(d / 'F.npy')
            self.Fneu  = np.load(d / 'Fneu.npy')
            ic         = np.load(d / 'iscell.npy')
            self.ops   = np.load(d / 'ops.npy',    allow_pickle=True).item()

            self.n_rois = len(self.stat)
            self.iscell = ic[:, 0].astype(bool)
            self.Ly     = int(self.ops['Ly'])
            self.Lx     = int(self.ops['Lx'])

            # deterministic random colour per ROI
            rng  = np.random.default_rng(42)
            hues = rng.uniform(0.0, 1.0, self.n_rois)
            self._colors = np.array(
                [_hsv8(h, 0.85, 0.95) for h in hues], dtype=np.uint8)

            self._status('Computing boundaries…')
            QApplication.processEvents()
            self._build_boundaries()

            self._load_bg()
            self._lasso_polys.clear()
            self._lasso_mask = None
            self.current_roi = 0

            # enable controls
            self.spin.setRange(0, self.n_rois - 1)
            self.spin.setValue(0)
            self.lbl_of.setText(f'/ {self.n_rois - 1}')
            for w in (self.spin, self.chk, self.btn_save, self.btn_lasso,
                      self.btn_all, self.btn_none,
                      self.combo_bg, self.sld_min, self.sld_max):
                w.setEnabled(True)

            self._refresh_all()
            for vb in (self.vb_left, self.vb_right):
                vb.setRange(xRange=[0, self.Lx], yRange=[0, self.Ly], padding=0)
            self._update_status()

        except Exception as exc:
            QMessageBox.critical(self, 'Load error', str(exc))
            import traceback
            traceback.print_exc()

    def _on_save(self):
        if self.plane_dir is None:
            return
        ic = np.load(self.plane_dir / 'iscell.npy')
        ic[:, 0] = self.iscell.astype(float)
        np.save(self.plane_dir / 'iscell.npy', ic)
        self._status(f'Saved  —  {int(self.iscell.sum())} / {self.n_rois} selected')

    # ─── Background helpers ───────────────────────────────────────────────────

    def _load_bg(self):
        self._bg_raw = {
            k: np.asarray(self.ops[k], dtype=float)
            for k in self._BG_KEYS if k in self.ops
        }
        for k in self._BG_KEYS:
            if k in self._bg_raw:
                self._bg_key = k
                break

        self.combo_bg.blockSignals(True)
        self.combo_bg.clear()
        for i, k in enumerate(self._BG_KEYS):
            self.combo_bg.addItem(
                self._BG_NAMES[i] + ('' if k in self._bg_raw else ' (N/A)'))
        self.combo_bg.setCurrentIndex(self._BG_KEYS.index(self._bg_key))
        self.combo_bg.blockSignals(False)

    def _bg_normalised(self):
        """Return background image normalised to [0, 1] using percentile sliders."""
        if self._bg_key not in self._bg_raw:
            return np.zeros((self.Ly, self.Lx), dtype=float)
        img  = self._bg_raw[self._bg_key]
        pmin = self.sld_min.value() / 10.0   # slider 0–990 → 0–99th percentile
        pmax = self.sld_max.value() / 10.0   # slider 10–1000 → 1–100th percentile
        lo, hi = np.percentile(img, pmin), np.percentile(img, pmax)
        if hi <= lo:
            hi = lo + 1e-6
        return np.clip((img - lo) / (hi - lo), 0.0, 1.0)

    # ─── Boundary & ROI-map precompute ────────────────────────────────────────

    def _build_boundaries(self):
        """
        For each ROI, precompute the set of boundary pixels (pixels that have
        at least one 4-connected neighbour not in the ROI).
        Also build _roi_map: pixel → ROI index lookup for click detection.
        """
        self._bnd_y  = []
        self._bnd_x  = []
        self._roi_map = np.full((self.Ly, self.Lx), -1, dtype=np.int32)

        for i, roi in enumerate(self.stat):
            yp = np.asarray(roi['ypix'], dtype=int)
            xp = np.asarray(roi['xpix'], dtype=int)

            # clip to image bounds
            ok = (yp >= 0) & (yp < self.Ly) & (xp >= 0) & (xp < self.Lx)
            yp, xp = yp[ok], xp[ok]

            # roi_map: last-writer wins (acceptable for overlapping ROIs)
            self._roi_map[yp, xp] = i

            # boundary via set lookup
            ps = set(zip(yp.tolist(), xp.tolist()))
            by, bx = [], []
            for y, x in ps:
                if ((y - 1, x) not in ps or (y + 1, x) not in ps or
                        (y, x - 1) not in ps or (y, x + 1) not in ps):
                    by.append(y)
                    bx.append(x)
            if not by:                  # single-pixel or fully interior fallback
                by, bx = yp.tolist(), xp.tolist()

            self._bnd_y.append(np.array(by, dtype=np.int32))
            self._bnd_x.append(np.array(bx, dtype=np.int32))

    # ─── Overlay building ─────────────────────────────────────────────────────

    def _roi_overlay(self, sel_side):
        """
        Build an RGBA (Ly, Lx, 4) image with:
          • Boundary outlines for all ROIs on the given side (selected or not).
          • Full fill (semi-transparent) for the currently active ROI.
        """
        ov = np.zeros((self.Ly, self.Lx, 4), dtype=np.uint8)

        for i in range(self.n_rois):
            if bool(self.iscell[i]) != sel_side:
                continue
            c = self._colors[i]
            ov[self._bnd_y[i], self._bnd_x[i], :3] = c
            ov[self._bnd_y[i], self._bnd_x[i],  3] = 220

        # highlight active ROI with a filled semi-transparent mask
        cur = self.current_roi
        if 0 <= cur < self.n_rois and bool(self.iscell[cur]) == sel_side:
            roi = self.stat[cur]
            fy  = np.clip(roi['ypix'].astype(int), 0, self.Ly - 1)
            fx  = np.clip(roi['xpix'].astype(int), 0, self.Lx - 1)
            c   = self._colors[cur]
            ov[fy, fx, :3] = c
            ov[fy, fx,  3] = 155

        return ov

    def _lasso_overlay(self):
        """Build a dark semi-transparent RGBA overlay for pixels outside the lasso area."""
        ov = np.zeros((self.Ly, self.Lx, 4), dtype=np.uint8)
        if self._lasso_mask is not None:
            out = ~self._lasso_mask
            ov[out, :3] = 20
            ov[out,  3] = 145
        return ov

    # ─── Refresh ──────────────────────────────────────────────────────────────

    def _refresh_images(self):
        bg8    = (self._bg_normalised() * 255).astype(np.uint8)
        bg_rgb = np.stack([bg8, bg8, bg8], axis=-1)
        lass   = self._lasso_overlay()

        self.img_bg_l.setImage(bg_rgb)
        self.img_bg_r.setImage(bg_rgb.copy())
        self.img_roi_l.setImage(self._roi_overlay(True))
        self.img_roi_r.setImage(self._roi_overlay(False))
        self.img_lasso_l.setImage(lass)
        self.img_lasso_r.setImage(lass.copy())

        n = int(self.iscell.sum())
        self.lbl_left.setText(f'Selected  ({n})')
        self.lbl_right.setText(f'Not selected  ({self.n_rois - n})')

    def _refresh_traces(self):
        self.plt_f.clear()
        self.plt_d.clear()

        if self.F is None or self.current_roi >= self.n_rois:
            return

        i    = self.current_roi
        t    = np.arange(self.F.shape[1])
        f    = self.F[i]
        fneu = self.Fneu[i]

        self.plt_f.plot(t, f,       pen=pg.mkPen('#5b9bd5', width=1), name='F')
        self.plt_f.plot(t, fneu,    pen=pg.mkPen('#d55b5b', width=1), name='Fneu')
        self.plt_d.plot(t, f - fneu, pen=pg.mkPen('#5bd58e', width=1))

        self.plt_f.setTitle(f'ROI {i}   —   F (blue)  &  Fneu (red)')
        self.plt_d.setTitle(f'ROI {i}   —   F − Fneu')

    def _refresh_spinner(self):
        self.spin.blockSignals(True)
        self.spin.setValue(self.current_roi)
        self.spin.blockSignals(False)

        self.chk.blockSignals(True)
        self.chk.setChecked(bool(self.iscell[self.current_roi]))
        self.chk.blockSignals(False)

    def _refresh_all(self):
        self._refresh_images()
        self._refresh_traces()
        self._refresh_spinner()

    # ─── Slots ────────────────────────────────────────────────────────────────

    def _on_bg_changed(self, idx):
        k = self._BG_KEYS[idx]
        if k in self._bg_raw:
            self._bg_key = k
            self._refresh_images()

    def _on_sat(self):
        if self.sld_min.value() < self.sld_max.value():
            self._refresh_images()

    def _on_spin(self, val):
        if self.stat is None:
            return
        self.current_roi = val
        self._refresh_spinner()
        self._refresh_traces()
        self._refresh_images()

    def _on_toggle(self, state):
        if self.stat is None:
            return
        self.iscell[self.current_roi] = bool(state)
        self._refresh_images()
        self._update_status()

    def _on_img_click(self, col, row):
        """Jump to whichever ROI owns the clicked pixel (if any)."""
        if self._roi_map is None:
            return
        r   = int(np.clip(row, 0, self.Ly - 1))
        c   = int(np.clip(col, 0, self.Lx - 1))
        idx = int(self._roi_map[r, c])
        if idx >= 0:
            self.spin.setValue(idx)          # triggers _on_spin

    def _bulk(self, on):
        self.iscell[:] = on
        self._refresh_all()
        self._update_status()

    # ─── Lasso ────────────────────────────────────────────────────────────────

    def _on_lasso_toggle(self, checked):
        self.vb_left.set_draw_mode(checked)
        self.vb_right.set_draw_mode(checked)

    def _on_lasso_complete(self, pts):
        """Called when the user releases the mouse after freehand drawing."""
        # turn off draw mode regardless of what happens next
        self.btn_lasso.setChecked(False)
        self.vb_left.set_draw_mode(False)
        self.vb_right.set_draw_mode(False)

        if self._lasso_polys:
            msg     = QMessageBox(self)
            msg.setWindowTitle('Area already defined')
            msg.setText('An area selection already exists.\n'
                        'Replace it with the new outline, or add to it?')
            b_rep   = msg.addButton('Replace',   QMessageBox.DestructiveRole)
            b_add   = msg.addButton('Add to it', QMessageBox.AcceptRole)
            msg.addButton('Cancel', QMessageBox.RejectRole)
            msg.exec_()
            clicked = msg.clickedButton()
            if clicked is b_rep:
                self._lasso_polys.clear()
            elif clicked is not b_add:
                return                       # Cancel: discard new polygon

        self._lasso_polys.append(pts)
        self._compute_lasso_mask()
        self._apply_lasso_deselect()
        self.btn_clear_lasso.setEnabled(True)
        self._refresh_all()
        self._update_status()

    def _on_clear_lasso(self):
        self._lasso_polys.clear()
        self._lasso_mask = None
        self.btn_clear_lasso.setEnabled(False)
        self._refresh_images()

    def _compute_lasso_mask(self):
        """
        Union of all drawn polygons → boolean (Ly, Lx) mask.
        Polygon points are in viewbox (col, row) = (x, y) coordinates.
        """
        mask = np.zeros((self.Ly, self.Lx), dtype=bool)

        # Build a (Ly*Lx, 2) query grid in (col, row) space
        rows, cols = np.mgrid[0:self.Ly, 0:self.Lx]
        query = np.column_stack([cols.ravel().astype(float),
                                 rows.ravel().astype(float)])

        for poly in self._lasso_polys:
            closed = np.vstack([poly, poly[0]])
            inside = MplPath(closed).contains_points(query)
            mask  |= inside.reshape(self.Ly, self.Lx)

        self._lasso_mask = mask

    def _apply_lasso_deselect(self):
        """Deselect any ROI whose centroid falls outside the current lasso mask."""
        if self._lasso_mask is None:
            return
        for i, roi in enumerate(self.stat):
            med = roi['med']                            # [mean_ypix, mean_xpix]
            r   = int(np.clip(round(med[0]), 0, self.Ly - 1))
            c   = int(np.clip(round(med[1]), 0, self.Lx - 1))
            if not self._lasso_mask[r, c]:
                self.iscell[i] = False

    # ─── Keyboard ─────────────────────────────────────────────────────────────

    def keyPressEvent(self, ev):
        if self.stat is None:
            super().keyPressEvent(ev)
            return
        k = ev.key()
        if k == QtCore.Qt.Key_Right:
            self.spin.setValue(min(self.current_roi + 1, self.n_rois - 1))
        elif k == QtCore.Qt.Key_Left:
            self.spin.setValue(max(self.current_roi - 1, 0))
        elif k == QtCore.Qt.Key_Space:
            self.chk.setChecked(not self.chk.isChecked())
        else:
            super().keyPressEvent(ev)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _update_status(self):
        if self.plane_dir is not None:
            n = int(self.iscell.sum())
            self._status(
                f'{self.plane_dir.name}  ·  {self.n_rois} ROIs  ·  {n} selected')

    def _status(self, txt):
        self.lbl_status.setText(txt)


# ─── Module-level helpers ─────────────────────────────────────────────────────

def _vsep():
    """Thin vertical separator for the toolbar."""
    f = QFrame()
    f.setFrameShape(QFrame.VLine)
    f.setFrameShadow(QFrame.Sunken)
    return f


def _slider(lo, hi, val):
    s = QSlider(QtCore.Qt.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    s.setFixedWidth(100)
    return s


def _hsv8(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    plane_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    win = ManualROISelector(plane_dir)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
