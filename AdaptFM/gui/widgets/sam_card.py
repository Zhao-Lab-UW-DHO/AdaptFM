# AdaptFM/gui/widgets/sam_card.py
"""
Self-contained card for a SAM model that installs as a package into the
main AdaptFM environment (not a separate conda env). Deliberately bypasses
ENV_REGISTRY / probe_all / _on_action — there is no conda env here, and it
should never be able to reach a `conda env remove` code path.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
from typing import Callable, Optional
import sys
from qtpy.QtCore import Qt, Signal, QProcess
from qtpy.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QMessageBox, QSizePolicy,
)

_INSTALLED_COLOR   = "#4caf50"
_UNINSTALLED_COLOR = "#9e9e9e"
_WARNING_COLOR     = "#f44336"
_UPDATE_COLOR      = "#ff9800"
_CARD_BG_DARK      = "#2b2b2b"
_CARD_BORDER       = "#444"


class SamCard(QFrame):
    pytorch_config_clicked = Signal()
    probeFinished = Signal(bool)

    def __init__(
        self,
        *,
        key: str,
        display_name: str,
        description: str,
        import_name: str,        # e.g. "sam2" — used to detect install
        install_command: str,    # e.g. "adaptfm-install-sam2"
        uninstall_command: str,  # e.g. "adaptfm-uninstall-sam2"
        requires_pytorch: bool,
        log_fn: Callable[..., None],
        run_process_fn: Callable[..., None],
        parent=None,
        source_dir
    ):
        super().__init__(parent)
        self._key = key
        self._display_name = display_name
        self._description = description
        self._import_name = import_name
        self._install_command = install_command
        self._uninstall_command = uninstall_command
        self._requires_pytorch = requires_pytorch
        self._log = log_fn
        self._run_process = run_process_fn
        self._source_dir = source_dir

        self._pending_op = None  # "install" or "uninstall"
        self._installed = self._probe()
        self._build()

    # ------------------------------------------------------------------
    def _probe(self) -> bool:
        """Check the *current* interpreter — there is no other env to check."""
        return importlib.util.find_spec(self._import_name) is not None

    # ------------------------------------------------------------------
    def _build(self):
        self.setObjectName("samCard")
        self.setStyleSheet(f"""
            QFrame#samCard {{
                background: {_CARD_BG_DARK};
                border: 1px solid {_CARD_BORDER};
                border-radius: 6px;
            }}
        """)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 10)
        outer.setSpacing(4)

        hdr = QHBoxLayout()
        title = QLabel(self._display_name)
        title.setStyleSheet("font-size: 14px; font-weight: 700; color: #e8e8e8;")
        hdr.addWidget(title)
        hdr.addStretch()
        self._badge = QLabel()
        hdr.addWidget(self._badge)
        outer.addLayout(hdr)

        desc = QLabel(self._description)
        desc.setStyleSheet("color: #999; font-size: 11px;")
        desc.setWordWrap(True)
        outer.addWidget(desc)

        if self._requires_pytorch:
            from AdaptFM.gui.widgets.pytorch_config_widget import PYTORCH_CMD_FILE
            if PYTORCH_CMD_FILE.exists() and PYTORCH_CMD_FILE.read_text().strip():
                warn_text = (
                    '⚠  Uses a custom PyTorch build  '
                    f'<a href="configure" style="color:{_UPDATE_COLOR}; font-size:10px;">change</a>'
                )
            else:
                warn_text = (
                    f'<span style="color:{_WARNING_COLOR};">⚠  PyTorch not configured</span>  '
                    f'<a href="configure" style="color:{_UPDATE_COLOR}; font-size:10px;">configure now ↑</a>'
                )
            warn = QLabel(warn_text)
            warn.setOpenExternalLinks(False)
            warn.setTextInteractionFlags(Qt.TextBrowserInteraction)
            warn.linkActivated.connect(lambda _: self.pytorch_config_clicked.emit())
            warn.setStyleSheet("font-size: 10px;")
            outer.addWidget(warn)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._action_btn = QPushButton()
        self._action_btn.setFixedWidth(90)
        self._action_btn.clicked.connect(self._on_action_clicked)
        btn_row.addWidget(self._action_btn)
        outer.addLayout(btn_row)

        self._refresh_badge()

    # ------------------------------------------------------------------
    def _refresh_badge(self):
        if self._installed:
            self._badge.setText("● Installed")
            self._badge.setStyleSheet(f"color: {_INSTALLED_COLOR}; font-size: 11px; font-weight: 600;")
            self._action_btn.setText("Uninstall")
            self._action_btn.setStyleSheet(
                "background: #555; color: #fff; border-radius: 4px; padding: 4px 10px;"
            )
        else:
            self._badge.setText("○ Not installed")
            self._badge.setStyleSheet(f"color: {_UNINSTALLED_COLOR}; font-size: 11px; font-weight: 600;")
            self._action_btn.setText("Install")
            self._action_btn.setStyleSheet(
                f"background: {_INSTALLED_COLOR}; color: #000; border-radius: 4px;"
                f" padding: 4px 10px; font-weight: 600;"
            )

    def _on_action_clicked(self):
        if self._installed:
            self._on_uninstall_clicked()
        else:
            self._on_install_clicked()

# Helper to reliably resolve commands in the current Python environment
    def _find_exe(self, cmd: str) -> str | None:
        exe = shutil.which(cmd)
        if exe:
            return exe
        
        # Fallback to current Python interpreter's bin/Scripts directory
        from pathlib import Path
        bin_dir = Path(sys.executable).parent
        candidate = bin_dir / cmd
        if candidate.exists():
            return str(candidate)
        if sys.platform == "win32":
            candidate_exe = bin_dir / f"{cmd}.exe"
            if candidate_exe.exists():
                return str(candidate_exe)
        return None

    def _on_install_clicked(self):
        exe = self._find_exe(self._install_command)
        if not exe:
            self._log(
                f"[error] Command '{self._install_command}' not found on PATH or in environment. "
                f"Did you run `pip install -e .`?", color=_WARNING_COLOR,
            )
            return
        
        self.set_busy(True)  # Lock UI during execution
        self._log(f"\n▶ {self._install_command}", bold=True)
        self._pending_op = "install"
        self._run_process(exe, [], label=self._display_name, on_done=self._on_op_done)

    def _on_uninstall_clicked(self):
        reply = QMessageBox.question(
            self, "Confirm uninstall",
            f"Uninstall {self._display_name}?\n\n"
            f"This removes the package (and its cloned source checkpoint) from "
            f"the AdaptFM environment. No conda environment is affected.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        
        exe = self._find_exe(self._uninstall_command)
        if not exe:
            self._log(f"[error] Command '{self._uninstall_command}' not found on PATH or in environment.", color=_WARNING_COLOR)
            return
        
        self.set_busy(True)  # Lock UI during execution
        self._log(f"\n▶ {self._uninstall_command}", bold=True)
        self._pending_op = "uninstall"
        self._run_process(exe, [], label=self._display_name, on_done=self._on_op_done)


    # def _on_op_done(self, exit_code: int):
    #     import importlib
    #     importlib.invalidate_caches()  # Flush Python's module import cache

    #     if exit_code == 0:
    #         self._log("\n✓ Done (exit 0)", color=_INSTALLED_COLOR, bold=True)

    #         if self._pending_op == "install":
    #             self._installed = True
    #         elif self._pending_op == "uninstall":
    #             self._installed = False

    #     else:
    #         self._log(f"\n✗ Exited with code {exit_code}", color=_WARNING_COLOR, bold=True)

    #     self._pending_op = None
    #     self.set_busy(False)  # <--- Re-enable widget interaction
    #     self._refresh_badge()

    def _resolve_package_root(self) -> Optional[Path]:
        """Handle both flat (<repo>/sam2/) and src-layout (<repo>/src/sam2/) checkouts."""
        if self._source_dir is None:
            return None
        if (self._source_dir / self._import_name).exists():
            return self._source_dir
        src_layout = self._source_dir / "src"
        if (src_layout / self._import_name).exists():
            return src_layout
        return None

    def _on_op_done(self, exit_code: int):
        self.set_busy(False)

        if exit_code == 0:
            self._log("\n✓ Done (exit 0)", color=_INSTALLED_COLOR, bold=True)
            pkg_root = self._resolve_package_root()
            path_entry = str(pkg_root) if pkg_root else None

            if self._pending_op == "install" and path_entry:
                if path_entry not in sys.path:
                    sys.path.insert(0, path_entry)
            elif self._pending_op == "uninstall" and path_entry:
                if path_entry in sys.path:
                    sys.path.remove(path_entry)
                sys.modules.pop(self._import_name, None)

            importlib.invalidate_caches()
            # Trust a live probe rather than the exit code, so the badge
            # reflects whether it's actually importable right now.
            self._installed = importlib.util.find_spec(self._import_name) is not None

            if self._pending_op == "install" and not self._installed:
                self._log(
                    f"[warn] {self._display_name} installed but not yet importable — "
                    f"a restart of AdaptFM may be required.",
                    color=_WARNING_COLOR,
                )
        else:
            self._log(f"\n✗ Exited with code {exit_code}", color=_WARNING_COLOR, bold=True)

        self._pending_op = None
        self._refresh_badge()


    def set_busy(self, busy: bool):
        self.setEnabled(not busy)