
"""
env_manager_dialog.py
---------------------
Napari "Environments" menu → opens this QDialog.

Layout
------
  ┌─ EnvironmentManagerDialog ──────────────────────────────────────────┐
  │  [↻ Check for updates]                                              │
  │                                                                     │
  │  ▶ PyTorch Configuration  ● Configured   <- collapsible card       │
  │    (expands to show pip command editor + Save / Save & Install)     │
  │                                                                     │
  │  ┌─ EnvCard: CellposeSAM ──────────────────────── ● Installed ─┐  │
  │  │  Cellpose segmentation with SAM backbone.                    │  │
  │  │  ⚠ Requires PyTorch — click to configure  <- clickable link  │  │
  │  │  cellpose  1.0.2  →  1.1.0  [Update]                        │  │
  │  │  PyTorch   2.2.0  ✓  up to date                             │  │
  │  │                          [Uninstall]                         │  │
  │  └──────────────────────────────────────────────────────────────┘  │
  │                                                                     │
  │  ┌─ EnvCard: SAM 2 ─────────────────────────── ○ Not installed ─┐ │
  │  │  ...                                        [Install]         │ │
  │  └──────────────────────────────────────────────────────────────┘ │
  │                                                                     │
  │  ── Log ──────────────────────────────────────────────────────────  │
  │  [live output from install / uninstall / update commands]          │
  └─────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Callable, Optional

from qtpy.QtCore import (
    Qt, QProcess, QThread, Signal, QObject,QProcessEnvironment
)
from qtpy.QtGui import QFont, QTextCursor
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QFrame, QTextEdit, QSizePolicy,
    QMessageBox, QProgressBar, QSplitter,QInputDialog
)

from AdaptFM.install.env_registry import ENV_REGISTRY, EnvironmentSpec
from AdaptFM.install.env_inspector import (
    EnvStatus, PackageVersionInfo, probe_all,
)
from AdaptFM.gui.widgets.pytorch_config_widget import PyTorchConfigWidget
from AdaptFM.gui.widgets.sam_card import SamCard
from AdaptFM.gui.widgets.ssvt_card import SSVTCard
from AdaptFM.model.model_utils import REPO_ROOT

# ---------------------------------------------------------------------------
# Colour / style constants (kept minimal so they work on both light & dark Qt)
# ---------------------------------------------------------------------------

_INSTALLED_COLOR   = "#4caf50"   # green
_UNINSTALLED_COLOR = "#9e9e9e"   # grey
_UPDATE_COLOR      = "#ff9800"   # amber
_WARNING_COLOR     = "#f44336"   # red-ish
_CARD_BG_DARK      = "#2b2b2b"
_CARD_BORDER       = "#444"
_MONO_FONT         = "Consolas, 'Courier New', monospace"


# ---------------------------------------------------------------------------
# Background probe worker
# ---------------------------------------------------------------------------

class _ProbeWorker(QObject):
    finished = Signal(list)   # list[EnvStatus]
    error    = Signal(str)

    def __init__(self, specs):
        super().__init__()
        self._specs = specs

    def run(self):
        try:
            results = probe_all(self._specs)
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Per-package version row
# ---------------------------------------------------------------------------

class _PackageRow(QWidget):
    update_requested = Signal(str)   # emits the pip package import_name

    def __init__(self, info: PackageVersionInfo, parent=None):
        super().__init__(parent)
        self._info = info
        self._build()

    def _build(self):
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(8)

        # Package name
        name_lbl = QLabel(self._info.spec.display_name)
        name_lbl.setFixedWidth(110)
        name_lbl.setStyleSheet("font-weight: 600; color: #ccc;")
        row.addWidget(name_lbl)

        installed = self._info.installed_version or "—"
        latest    = self._info.latest_version

        inst_lbl = QLabel(f"installed: {installed}")
        inst_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        row.addWidget(inst_lbl)

        if self._info.update_available and latest:
            arrow_lbl = QLabel(f"→  {latest} available")
            arrow_lbl.setStyleSheet(f"color: {_UPDATE_COLOR}; font-size: 11px;")
            row.addWidget(arrow_lbl)

            upd_btn = QPushButton("Update")
            upd_btn.setFixedWidth(70)
            upd_btn.setStyleSheet(
                f"background: {_UPDATE_COLOR}; color: #000; border-radius: 4px;"
                f" padding: 2px 6px; font-size: 11px;"
            )
            upd_btn.clicked.connect(
                lambda: self.update_requested.emit(self._info.spec.import_name)
            )
            row.addWidget(upd_btn)
        elif latest and not self._info.error:
            ok_lbl = QLabel("✓ up to date")
            ok_lbl.setStyleSheet(f"color: {_INSTALLED_COLOR}; font-size: 11px;")
            row.addWidget(ok_lbl)
        elif self._info.error:
            err_lbl = QLabel("⚠ version check failed")
            err_lbl.setStyleSheet(f"color: {_WARNING_COLOR}; font-size: 11px;")
            err_lbl.setToolTip(self._info.error)
            row.addWidget(err_lbl)
        elif self._info.spec.update_source == "none":
            pass  # intentionally silent
        else:
            chk_lbl = QLabel("checking…")
            chk_lbl.setStyleSheet("color: #777; font-size: 11px;")
            row.addWidget(chk_lbl)

        row.addStretch()


# ---------------------------------------------------------------------------
# Per-environment card
# ---------------------------------------------------------------------------

class _EnvCard(QFrame):
    """One card per EnvironmentSpec."""

    action_requested       = Signal(str, str)  # (env_key, action)
    pytorch_config_clicked = Signal()           # user clicked the PyTorch warning link

    def __init__(self, status: EnvStatus, parent=None):
        super().__init__(parent)
        self._status = status
        self._pkg_rows: list[_PackageRow] = []
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        spec = self._status.spec

        self.setObjectName("envCard")
        self.setStyleSheet(f"""
            QFrame#envCard {{
                background: {_CARD_BG_DARK};
                border: 1px solid {_CARD_BORDER};
                border-radius: 6px;
            }}
        """)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 10)
        outer.setSpacing(4)

        # --- Header row ---
        hdr = QHBoxLayout()
        hdr.setSpacing(8)

        title = QLabel(spec.display_name)
        title.setStyleSheet("font-size: 14px; font-weight: 700; color: #e8e8e8;")
        hdr.addWidget(title)

        hdr.addStretch()

        # Status badge
        if self._status.is_installed:
            badge_text  = "● Installed"
            badge_color = _INSTALLED_COLOR
        else:
            badge_text  = "○ Not installed"
            badge_color = _UNINSTALLED_COLOR

        badge = QLabel(badge_text)
        badge.setStyleSheet(
            f"color: {badge_color}; font-size: 11px; font-weight: 600;"
        )
        hdr.addWidget(badge)
        outer.addLayout(hdr)

        # --- Description ---
        desc = QLabel(spec.description)
        desc.setStyleSheet("color: #999; font-size: 11px;")
        desc.setWordWrap(True)
        outer.addWidget(desc)

        # --- PyTorch swap warning (clickable link → opens config card) ---
        if spec.requires_pytorch_swap:
            from AdaptFM.gui.widgets.pytorch_config_widget import PYTORCH_CMD_FILE
            if PYTORCH_CMD_FILE.exists() and PYTORCH_CMD_FILE.read_text().strip():
                warn_text = (
                    '⚠  Uses a custom PyTorch build  '
                    f'<a href="configure" style="color:{_UPDATE_COLOR}; font-size:10px;">'
                    'change</a>'
                )
            else:
                warn_text = (
                    f'<span style="color:{_WARNING_COLOR};">⚠  PyTorch not configured</span>  '
                    f'<a href="configure" style="color:{_UPDATE_COLOR}; font-size:10px;">'
                    'configure now ↑</a>'
                )
            warn = QLabel(warn_text)
            warn.setOpenExternalLinks(False)
            warn.setTextInteractionFlags(Qt.TextBrowserInteraction)
            warn.linkActivated.connect(lambda _: self.pytorch_config_clicked.emit())
            warn.setStyleSheet("font-size: 10px;")
            outer.addWidget(warn)

        # --- Package version rows (only when installed) ---
        if self._status.is_installed and self._status.packages:
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet(f"color: {_CARD_BORDER};")
            outer.addWidget(sep)

            for pkg_info in self._status.packages:
                row = _PackageRow(pkg_info)
                row.update_requested.connect(self._on_update_pkg)
                self._pkg_rows.append(row)
                outer.addWidget(row)

        # --- Action buttons ---
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        if self._status.is_installed:
            uninst_btn = QPushButton("Uninstall")
            uninst_btn.setFixedWidth(90)
            uninst_btn.setStyleSheet(
                f"background: #555; color: #fff; border-radius: 4px; padding: 4px 10px;"
            )
            uninst_btn.clicked.connect(self._on_uninstall)
            btn_row.addWidget(uninst_btn)
        else:
            inst_btn = QPushButton("Install")
            inst_btn.setFixedWidth(90)
            inst_btn.setStyleSheet(
                f"background: {_INSTALLED_COLOR}; color: #000; border-radius: 4px;"
                f" padding: 4px 10px; font-weight: 600;"
            )
            inst_btn.clicked.connect(self._on_install)
            btn_row.addWidget(inst_btn)

        outer.addLayout(btn_row)

    # ------------------------------------------------------------------
    def _on_install(self):
        self.action_requested.emit(self._status.spec.key, "install")

    def _on_uninstall(self):
        spec = self._status.spec
        reply = QMessageBox.question(
            self, "Confirm uninstall",
            f"Remove conda environment '{spec.conda_env_name}'?\n\n"
            f"This will delete the environment and all its packages.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.action_requested.emit(spec.key, "uninstall")

    def _on_update_pkg(self, import_name: str):
        self.action_requested.emit(self._status.spec.key, f"update:{import_name}")

    # ------------------------------------------------------------------
    def set_busy(self, busy: bool):
        """Grey-out the card while an operation is running."""
        self.setEnabled(not busy)


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class EnvironmentManagerDialog(QDialog):
    """
    Launched from the Environments menu item.
    Probes all environments in a background thread, renders cards,
    and runs install/uninstall/update commands via QProcess (non-blocking).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AdaptFM — Environment Manager")
        self.resize(740, 640)
        self.setMinimumWidth(560)

        self._cards: dict[str, _EnvCard] = {}
        self._process: Optional[QProcess] = None
        self._probe_thread: Optional[QThread] = None
        self._probe_worker: Optional[_ProbeWorker] = None

        self._build_ui()
        self._start_probe()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # Top bar
        top = QHBoxLayout()
        title = QLabel("Manage Environments")
        title.setStyleSheet("font-size: 16px; font-weight: 700;")
        top.addWidget(title)
        top.addStretch()

        self._refresh_btn = QPushButton("↻  Check for updates")
        self._refresh_btn.setFixedWidth(160)
        self._refresh_btn.clicked.connect(self._start_probe)
        top.addWidget(self._refresh_btn)
        root.addLayout(top)

        # Status / spinner label
        self._status_lbl = QLabel("Scanning environments…")
        self._status_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        root.addWidget(self._status_lbl)

        # Progress bar (hidden when idle)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setFixedHeight(4)
        self._progress.setVisible(False)
        self._progress.setStyleSheet(
            "QProgressBar { border: none; background: #333; }"
            "QProgressBar::chunk { background: #4caf50; }"
        )
        root.addWidget(self._progress)

        # Splitter: cards (top) + log (bottom)
        splitter = QSplitter(Qt.Vertical)
        root.addWidget(splitter)

        # Card scroll area
        scroll_outer = QWidget()
        scroll_layout = QVBoxLayout(scroll_outer)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._card_container = QWidget()
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setContentsMargins(0, 0, 4, 0)
        self._card_layout.setSpacing(8)

        self._scroll.setWidget(self._card_container)
        scroll_layout.addWidget(self._scroll)
        splitter.addWidget(scroll_outer)


        # Log panel
        log_outer = QWidget()
        log_layout = QVBoxLayout(log_outer)
        log_layout.setContentsMargins(0, 4, 0, 0)

        log_lbl = QLabel("Install log")
        log_lbl.setStyleSheet("color: #888; font-size: 10px; font-weight: 600;")
        log_layout.addWidget(log_lbl)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setStyleSheet(
            "background: #1a1a1a; color: #d4d4d4; border: 1px solid #333;"
            " border-radius: 4px;"
        )
        self._log.setMinimumHeight(120)
        log_layout.addWidget(self._log)

        clear_btn = QPushButton("Clear log")
        clear_btn.setFixedWidth(80)
        clear_btn.setStyleSheet("color: #777; font-size: 10px;")
        clear_btn.clicked.connect(self._log.clear)
        log_layout.addWidget(clear_btn, alignment=Qt.AlignRight)
        splitter.addWidget(log_outer)

        splitter.setSizes([420, 200])

    # ------------------------------------------------------------------
    # Background probe
    # ------------------------------------------------------------------

    def _start_probe(self):
        # Kill any in-flight probe
        if self._probe_thread and self._probe_thread.isRunning():
            return

        self._status_lbl.setText("Scanning environments…")
        self._progress.setVisible(True)
        self._refresh_btn.setEnabled(False)
        self._clear_cards()

        self._probe_thread = QThread()
        self._probe_worker = _ProbeWorker(ENV_REGISTRY)
        self._probe_worker.moveToThread(self._probe_thread)
        self._probe_thread.started.connect(self._probe_worker.run)
        self._probe_worker.finished.connect(self._on_probe_done)
        self._probe_worker.error.connect(self._on_probe_error)
        self._probe_worker.finished.connect(self._probe_thread.quit)
        self._probe_worker.error.connect(self._probe_thread.quit)
        self._probe_thread.start()

    def _clear_cards(self):
        for card in self._cards.values():
            self._card_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

    def _on_probe_done(self, statuses: list):
        self._progress.setVisible(False)
        self._refresh_btn.setEnabled(True)

        installed = sum(1 for s in statuses if s.is_installed)
        total     = len(statuses)
        updates   = sum(
            1 for s in statuses
            for p in s.packages
            if p.update_available
        )

        parts = [f"{installed}/{total} environments installed"]
        if updates:
            parts.append(f"{updates} update{'s' if updates > 1 else ''} available")
        self._status_lbl.setText(" · ".join(parts))

        self._pytorch_card = PyTorchConfigWidget(
            log_fn=self._log_line,
            run_process_fn=self._run_process,
        )
        self._pytorch_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._card_layout.addWidget(self._pytorch_card)

        self._sam2_card = SamCard(
            key="SAM2", display_name="SAM 2",
            description="Segment Anything Model 2 (Meta). Installs directly into the AdaptFM environment.",
            import_name="sam2",
            install_command="adaptfm-install-sam2",
            uninstall_command="adaptfm-uninstall-sam2",
            requires_pytorch=True,
            log_fn=self._log_line,
            run_process_fn=self._run_process,
            source_dir = REPO_ROOT/"segmentation"/"sam2"
        )
        self._sam3_card = SamCard(
            key="SAM3", display_name="SAM 3",
            description="Segment Anything Model 3 (Meta). Installs directly into the AdaptFM environment.",
            import_name="sam3",
            install_command="adaptfm-install-sam3",
            uninstall_command="adaptfm-uninstall-sam3",
            requires_pytorch=True,
            log_fn=self._log_line,
            run_process_fn=self._run_process,
            source_dir = REPO_ROOT/"segmentation"/"sam3"
        )

        self._ssvt_card = SSVTCard(
            key="SSVT",display_name="SSVT",
            description = "Downloads the SSVT model from Hugging face",
            import_name=None,
            install_command="adaptfm-install-ssvt",
            uninstall_command="adaptfm-uninstall-ssvt",
            requires_pytorch=False,
            log_fn=self._log_line,
            run_process_fn=self._run_process,
            source_dir =None
        )

        self._sam2_card.pytorch_config_clicked.connect(self._focus_pytorch_card)
        self._sam3_card.pytorch_config_clicked.connect(self._focus_pytorch_card)
        self._card_layout.addWidget(self._sam2_card)
        self._card_layout.addWidget(self._sam3_card)
        self._card_layout.addWidget(self._ssvt_card)

        self._aux_cards = [self._pytorch_card, self._sam2_card, self._sam3_card,
                           self._ssvt_card]

        # Stretch goes last, AFTER every fixed card. Dynamic env cards get
        # inserted just before it via `idx = self._card_layout.count() - 1`.
        self._card_layout.addStretch()


        for status in statuses:
            card = _EnvCard(status)
            card.action_requested.connect(self._on_action)
            card.pytorch_config_clicked.connect(self._focus_pytorch_card)
            self._cards[status.spec.key] = card
            # Insert before the trailing stretch
            idx = self._card_layout.count() - 1
            self._card_layout.insertWidget(idx, card)

    def _on_probe_error(self, msg: str):
        self._progress.setVisible(False)
        self._refresh_btn.setEnabled(True)
        self._status_lbl.setText(f"Scan failed: {msg}")
        self._log_line(f"[probe error] {msg}", color=_WARNING_COLOR)

    # ------------------------------------------------------------------
    # Shared QProcess runner — used by both env actions and PyTorchConfigWidget
    # ------------------------------------------------------------------

    def _run_process(
        self,
        program: str,
        args: list[str],
        label: str = "",
        on_done: Optional[Callable[[int], None]] = None,
    ):
        """
        Start *program* with *args* in a QProcess.
        stdout/stderr stream to the log panel.
        on_done(exit_code) is called when the process finishes.
        Blocks all cards while running.
        """
        if self._process and self._process.state() != QProcess.NotRunning:
            QMessageBox.warning(
                self, "Busy",
                "Another operation is already running.\n"
                "Please wait for it to finish.",
            )
            return

        self._set_all_cards_busy(True)
        self._progress.setVisible(True)
        for card in self._aux_cards:
            card.setEnabled(False)
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyRead.connect(self._on_process_output)

        def _done(code, _status):
            self._progress.setVisible(False)
            self._set_all_cards_busy(False)
            
            for card in getattr(self, "_aux_cards", []):
                card.setEnabled(True)

            if on_done:
                on_done(code)

        self._process.finished.connect(_done)
        self._process.start(program, args)

    def _focus_pytorch_card(self):
        """Expand the PyTorch config card and scroll to it."""
        self._pytorch_card.expand()
        self._scroll.ensureWidgetVisible(self._pytorch_card)

    # ------------------------------------------------------------------
    # Action dispatch — install / uninstall / update
    # ------------------------------------------------------------------

    def _on_action(self, env_key: str, action: str):
        from AdaptFM.install.env_registry import ENV_REGISTRY_MAP
        spec = ENV_REGISTRY_MAP.get(env_key)
        if not spec:
            return

        if self._process and self._process.state() != QProcess.NotRunning:
            QMessageBox.warning(
                self, "Busy",
                "Another operation is already running.\n"
                "Please wait for it to finish.",
            )
            return

        if action == "install":
            self._run_command(spec, spec.install_command, env_key)
        elif action == "uninstall":
            self._run_uninstall(spec, env_key)
        elif action.startswith("update:"):
            pkg_name = action.split(":", 1)[1]
            self._run_pip_update(spec, pkg_name, env_key)

    # ------------------------------------------------------------------
    # Per-action helpers — delegate to _run_process
    # ------------------------------------------------------------------
    def _run_command(self, spec: EnvironmentSpec, command: str, env_key: str):
        """Run a named entry-point command (install / uninstall scripts)."""
        exe = shutil.which(command)
        if not exe:
            self._log_line(
                f"[error] Command '{command}' not found on PATH. "
                f"Did you run `pip install -e .`?",
                color=_WARNING_COLOR,
            )
            return
        
        extra_env = None
        if env_key in ("CellSAM", "ThreeDCellComposer") and command == spec.install_command:
            access_token, ok = QInputDialog.getText(
                self,
                "DeepCell Access Token",
                "Enter your DeepCell access token:"
            )
            if not ok:
                self._log_line("[install cancelled] token entry cancelled.", color=_WARNING_COLOR)
                return

            access_token = access_token.strip()
            if not access_token:
                QMessageBox.warning(
                    self,
                    "Missing token",
                    "A DeepCell access token is required to install CellSAM and ThreeDCellComposer."
                )
                return

            extra_env = {"DEEPCELL_ACCESS_TOKEN": access_token}

        self._log_line(f"\n▶ {command}", bold=True)
        self._set_all_cards_busy(True)
        self._progress.setVisible(True)

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyRead.connect(self._on_process_output)
        self._process.finished.connect(
            lambda code, status: self._on_process_done(code, status, env_key)
        )

        if extra_env:
            env = QProcessEnvironment.systemEnvironment()
            for k, v in extra_env.items():
                env.insert(k, v)
            self._process.setProcessEnvironment(env)

        self._process.start(exe, [])


    def _run_uninstall(self, spec: EnvironmentSpec, env_key: str):
        """Uninstall = run uninstall script if it exists, else conda env remove."""
        cmd = spec.uninstall_command  # e.g. "adaptfm-uninstall"
        exe = shutil.which(cmd)

        if exe:
            # Show command in UI
            self._log_line(f"\n▶ {cmd} {spec.conda_env_name}", bold=True)
            self._set_all_cards_busy(True)
            self._progress.setVisible(True)

            # Prepare QProcess
            self._process = QProcess(self)
            self._process.setProcessChannelMode(QProcess.MergedChannels)
            self._process.readyRead.connect(self._on_process_output)
            self._process.finished.connect(
                lambda code, status: self._on_process_done(code, status, env_key)
            )

            # Run uninstall script WITH ARGUMENTS
            self._process.start(exe, [spec.conda_env_name])
            return

        # -------------------------
        # Fallback: conda env remove
        # -------------------------
        self._log_line(
            f"\n▶ conda env remove -n {spec.conda_env_name} -y", bold=True
        )
        self._set_all_cards_busy(True)
        self._progress.setVisible(True)

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyRead.connect(self._on_process_output)
        self._process.finished.connect(
            lambda code, status: self._on_process_done(code, status, env_key)
        )
        self._process.start("conda", ["env", "remove", "-n", spec.conda_env_name, "-y"])

    def _run_pip_update(self, spec: EnvironmentSpec, import_name: str, env_key: str):
        """Run `conda run -n <env> pip install --upgrade <package>`."""
        args = ["run", "-n", spec.conda_env_name, "--no-capture-output",
                "pip", "install", "--upgrade", import_name]
        self._log_line(f"\n▶ conda {' '.join(args)}", bold=True)
        self._set_all_cards_busy(True)
        self._progress.setVisible(True)

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyRead.connect(self._on_process_output)
        self._process.finished.connect(
            lambda code, status: self._on_process_done(code, status, env_key)
        )
        self._process.start("conda", args)

    # ------------------------------------------------------------------
    # QProcess callbacks
    # ------------------------------------------------------------------

    def _on_process_output(self):
        data = self._process.readAll().data()
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = str(data)
        self._log_append(text)

    def _on_process_done(self, exit_code: int, _exit_status, env_key: str):
        self._progress.setVisible(False)
        self._set_all_cards_busy(False)

        if exit_code == 0:
            self._log_line(f"\n✓ Done (exit 0)", color=_INSTALLED_COLOR, bold=True)
        else:
            self._log_line(
                f"\n✗ Exited with code {exit_code}", color=_WARNING_COLOR, bold=True
            )

        # Re-probe just this environment so the card updates immediately
        self._reprobe_one(env_key)

    # ------------------------------------------------------------------
    # Targeted re-probe after an action completes
    # ------------------------------------------------------------------

    def _reprobe_one(self, env_key: str):
        from AdaptFM.install.env_registry import ENV_REGISTRY_MAP
        from AdaptFM.install.env_inspector import probe_env
        spec = ENV_REGISTRY_MAP.get(env_key)
        if not spec:
            return

        self._status_lbl.setText("Refreshing…")
        self._progress.setVisible(True)

        thread = QThread(self)
        worker = _SingleProbeWorker(spec)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(lambda status: self._on_single_probe_done(status, thread))
        worker.finished.connect(thread.quit)
        thread.start()
        # Keep a reference so it isn't GC'd
        self._last_single_thread = thread
        self._last_single_worker = worker

    def _on_single_probe_done(self, status, thread):
        self._progress.setVisible(False)
        self._status_lbl.setText("Ready.")

        env_key = status.spec.key
        old_card = self._cards.get(env_key)
        if old_card is None:
            return

        idx = self._card_layout.indexOf(old_card)
        old_card.deleteLater()

        new_card = _EnvCard(status)
        new_card.action_requested.connect(self._on_action)
        self._cards[env_key] = new_card
        self._card_layout.insertWidget(idx, new_card)

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    def _log_append(self, text: str):
        self._log.moveCursor(QTextCursor.End)
        self._log.insertPlainText(text)
        self._log.moveCursor(QTextCursor.End)

    def _log_line(self, text: str, color: str = "", bold: bool = False):
        self._log.moveCursor(QTextCursor.End)
        if color or bold:
            fmt_open  = ""
            fmt_close = ""
            if bold:
                fmt_open  += "<b>"
                fmt_close  = "</b>" + fmt_close
            if color:
                fmt_open  += f'<span style="color:{color};">'
                fmt_close  = "</span>" + fmt_close
            self._log.insertHtml(
                f"{fmt_open}{text.replace(chr(10), '<br>')}{fmt_close}<br>"
            )
        else:
            self._log.insertPlainText(text + "\n")
        self._log.moveCursor(QTextCursor.End)

    # ------------------------------------------------------------------

    def _set_all_cards_busy(self, busy: bool):
        for card in self._cards.values():
            card.set_busy(busy)

    def closeEvent(self, event):
        if self._process and self._process.state() != QProcess.NotRunning:
            reply = QMessageBox.question(
                self, "Operation in progress",
                "An install/uninstall is still running.\n"
                "Close anyway? (the process will keep running in the background)",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Lightweight single-env probe worker
# ---------------------------------------------------------------------------

class _SingleProbeWorker(QObject):
    finished = Signal(object)   # EnvStatus

    def __init__(self, spec):
        super().__init__()
        self._spec = spec

    def run(self):
        from AdaptFM.install.env_inspector import probe_env
        try:
            self.finished.emit(probe_env(self._spec))
        except Exception:
            pass
