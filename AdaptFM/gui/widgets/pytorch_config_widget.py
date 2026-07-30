"""
pytorch_config_widget.py
------------------------
Inline GUI replacement for `adaptfm-set-pytorch`.

Reads / writes  ~/.adaptfm/pytorch_cmd.txt  and, optionally, installs
PyTorch into the main AdaptFM conda environment via a QProcess so the
install log streams live into the parent dialog.

Designed to be embedded inside EnvironmentManagerDialog as a collapsible
section; it can also be used stand-alone.

Public API
----------
PyTorchConfigWidget(log_fn, run_process_fn, parent)
    log_fn(text, color, bold)  — write a line into the shared log panel
    run_process_fn(program, args, label, on_done)
                               — delegate QProcess execution to the dialog

Signals
-------
config_saved(cmd: str)   — emitted when the file is (re-)written
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Callable, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFrame, QSizePolicy, QToolButton,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_DIR       = Path.home() / ".adaptfm"
PYTORCH_CMD_FILE = CONFIG_DIR / "pytorch_cmd.txt"
ADAPTFM_ENV      = "AdaptFM"

_PYTORCH_URL = "https://pytorch.org/get-started/locally/"

_EXAMPLE_COMMANDS = [
    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
    "pip install torch torchvision torchaudio   # CPU-only",
]

# Style constants (match the rest of the dialog)
_CARD_BG    = "#2b2b2b"
_CARD_BD    = "#444"
_GREEN      = "#4caf50"
_AMBER      = "#ff9800"
_RED        = "#f44336"
_BLUE       = "#42a5f5"


def _read_saved_cmd() -> str:
    """Return the saved command, or '' if the file doesn't exist."""
    if PYTORCH_CMD_FILE.exists():
        return PYTORCH_CMD_FILE.read_text().strip()
    return ""


def _normalise_cmd(raw: str) -> str:
    """
    Accept anything the user pastes from pytorch.org and normalise it to a
    canonical  `pip install …`  string (without a leading pip3/pip).
    """
    s = raw.strip()
    for prefix in ("pip3 install ", "pip install "):
        if s.startswith(prefix):
            return "pip install " + s[len(prefix):]
    if s.startswith("install "):
        return "pip " + s
    # Bare package list → prepend pip install
    return "pip install " + s


def _validate_cmd(cmd: str) -> tuple[bool, str]:
    """
    Lightweight sanity check.  Returns (ok, error_message).
    We don't shell-execute anything here.
    """
    if not cmd.strip():
        return False, "Command is empty."
    try:
        parts = shlex.split(cmd)
    except ValueError as e:
        return False, f"Parse error: {e}"
    if len(parts) < 3:
        return False, "Too short — expected at least  pip install <package>."
    if parts[0] not in ("pip", "pip3"):
        return False, f"Expected command to start with 'pip' or 'pip3', got '{parts[0]}'."
    if parts[1] != "install":
        return False, f"Expected 'pip install …', got 'pip {parts[1]} …'."
    # Must contain at least one token that looks like a package name (not a flag)
    packages = [p for p in parts[2:] if not p.startswith("-")]
    if not packages:
        return False, "No package names found after 'pip install'."
    # Warn if it looks like torch is missing
    torch_tokens = {"torch", "pytorch"}
    if not any(t in p.lower() for p in packages for t in torch_tokens):
        return False, "Warning: 'torch' not found in the command — is this a PyTorch install?"
    return True, ""


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class PyTorchConfigWidget(QFrame):
    """
    Collapsible card that lets users set their PyTorch pip command.

    Parameters
    ----------
    log_fn:
        Callable(text, color="", bold=False) — writes into the shared log.
    run_process_fn:
        Callable(program: str, args: list[str], label: str,
                 on_done: Callable[[int], None])
        Delegates QProcess execution to the parent dialog so output streams
        to the shared log panel and the busy-lock is respected.
    """

    config_saved = Signal(str)   # emits the normalised command string

    def __init__(
        self,
        log_fn: Callable,
        run_process_fn: Callable,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._log        = log_fn
        self._run_proc   = run_process_fn
        self._expanded   = False
        self._build()
        self._load_saved()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self):
        self.setObjectName("pytorchCard")
        self.setStyleSheet(f"""
            QFrame#pytorchCard {{
                background: {_CARD_BG};
                border: 1px solid {_CARD_BD};
                border-radius: 6px;
            }}
        """)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(14, 8, 14, 8)
        self._outer.setSpacing(6)

        # ── Header (always visible) ─────────────────────────────────── #
        hdr = QHBoxLayout()
        hdr.setSpacing(8)

        self._toggle_btn = QToolButton()
        self._toggle_btn.setArrowType(Qt.RightArrow)
        self._toggle_btn.setStyleSheet("border: none; color: #aaa;")
        self._toggle_btn.clicked.connect(self._toggle)
        hdr.addWidget(self._toggle_btn)

        title = QLabel("PyTorch Configuration")
        title.setStyleSheet("font-size: 13px; font-weight: 700; color: #e8e8e8;")
        hdr.addWidget(title)

        hdr.addStretch()

        self._status_badge = QLabel()
        self._status_badge.setStyleSheet("font-size: 11px;")
        hdr.addWidget(self._status_badge)

        self._outer.addLayout(hdr)

        # ── Body (shown when expanded) ──────────────────────────────── #
        self._body = QWidget()
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(20, 4, 0, 4)
        body_layout.setSpacing(6)

        # Explanation
        intro = QLabel(
            "Paste the <b>pip install</b> command for your PyTorch build from "
            f"{_PYTORCH_URL}"
            "pytorch.org/get-started/locally</a>."
        )
        intro.setOpenExternalLinks(True)
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa; font-size: 11px;")
        body_layout.addWidget(intro)

        # Example hint
        eg = QLabel(
            f'<span style="color:#666; font-size:10px;">'
            f'e.g.  {_EXAMPLE_COMMANDS[0]}</span>'
        )
        eg.setWordWrap(True)
        body_layout.addWidget(eg)

        # Input row
        inp_row = QHBoxLayout()
        inp_row.setSpacing(6)

        self._cmd_edit = QLineEdit()
        self._cmd_edit.setPlaceholderText(
            "pip install torch torchvision torchaudio --index-url https://…"
        )
        self._cmd_edit.setStyleSheet(
            "background: #1e1e1e; color: #d4d4d4; border: 1px solid #555;"
            " border-radius: 4px; padding: 4px 8px; font-family: Consolas, monospace;"
            " font-size: 11px;"
        )
        self._cmd_edit.textChanged.connect(self._on_text_changed)
        inp_row.addWidget(self._cmd_edit)

        body_layout.addLayout(inp_row)

        # Validation message
        self._validation_lbl = QLabel("")
        self._validation_lbl.setStyleSheet("font-size: 10px;")
        self._validation_lbl.setWordWrap(True)
        body_layout.addWidget(self._validation_lbl)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()

        self._save_btn = QPushButton("Save")
        self._save_btn.setFixedWidth(80)
        self._save_btn.setEnabled(False)
        self._save_btn.setStyleSheet(
            f"background: {_GREEN}; color: #000; border-radius: 4px;"
            f" padding: 4px 10px; font-weight: 600;"
        )
        self._save_btn.clicked.connect(self._save)
        btn_row.addWidget(self._save_btn)

        self._install_btn = QPushButton("Save & install into AdaptFM env")
        self._install_btn.setEnabled(False)
        self._install_btn.setStyleSheet(
            f"background: {_BLUE}; color: #000; border-radius: 4px;"
            f" padding: 4px 10px; font-weight: 600; font-size: 11px;"
        )
        self._install_btn.clicked.connect(self._save_and_install)
        btn_row.addWidget(self._install_btn)

        body_layout.addLayout(btn_row)

        self._body.setVisible(False)
        self._outer.addWidget(self._body)

    # ------------------------------------------------------------------
    # Load saved state
    # ------------------------------------------------------------------

    def _load_saved(self):
        saved = _read_saved_cmd()
        if saved:
            self._cmd_edit.setText(saved)
            self._update_badge(configured=True)
        else:
            self._update_badge(configured=False)
            # Auto-expand if nothing is configured yet
            self._set_expanded(True)

    def _update_badge(self, configured: bool):
        if configured:
            self._status_badge.setText("● Configured")
            self._status_badge.setStyleSheet(
                f"color: {_GREEN}; font-size: 11px; font-weight: 600;"
            )
        else:
            self._status_badge.setText("○ Not configured")
            self._status_badge.setStyleSheet(
                f"color: {_AMBER}; font-size: 11px; font-weight: 600;"
            )

    # ------------------------------------------------------------------
    # Expand / collapse
    # ------------------------------------------------------------------

    def _toggle(self):
        self._set_expanded(not self._expanded)

    def _set_expanded(self, expanded: bool):
        self._expanded = expanded
        self._body.setVisible(expanded)
        self._toggle_btn.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow
        )

    def expand(self):
        """Called externally, e.g. from a warning badge click."""
        self._set_expanded(True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _on_text_changed(self, text: str):
        if not text.strip():
            self._validation_lbl.setText("")
            self._save_btn.setEnabled(False)
            self._install_btn.setEnabled(False)
            return

        normalised = _normalise_cmd(text)
        ok, msg = _validate_cmd(normalised)

        if ok:
            self._validation_lbl.setText("✓ Looks valid")
            self._validation_lbl.setStyleSheet(
                f"color: {_GREEN}; font-size: 10px;"
            )
            self._save_btn.setEnabled(True)
            self._install_btn.setEnabled(True)
        else:
            self._validation_lbl.setText(f"⚠  {msg}")
            self._validation_lbl.setStyleSheet(
                f"color: {_AMBER}; font-size: 10px;"
            )
            # Still allow saving if it's just a soft warning (starts with Warning:)
            is_soft = msg.startswith("Warning:")
            self._save_btn.setEnabled(is_soft)
            self._install_btn.setEnabled(is_soft)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save(self) -> str:
        """Write file, update badge, return the normalised command."""
        raw = self._cmd_edit.text().strip()
        normalised = _normalise_cmd(raw)
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        PYTORCH_CMD_FILE.write_text(normalised + "\n")
        self._update_badge(configured=True)
        self._log(
            f"✓ PyTorch command saved to {PYTORCH_CMD_FILE}",
            color=_GREEN,
            bold=True,
        )
        self.config_saved.emit(normalised)
        return normalised

    # ------------------------------------------------------------------
    # Save + install
    # ------------------------------------------------------------------

    def _save_and_install(self):
        cmd = self._save()
        parts = shlex.split(cmd)   # e.g. ["pip", "install", "torch", …]

        # Run inside AdaptFM conda env
        conda_args = [
            "run", "-n", ADAPTFM_ENV, "--no-capture-output",
        ] + parts

        self._log(
            f"\n▶ conda {' '.join(conda_args)}",
            bold=True,
        )

        self._run_proc(
            "conda",
            conda_args,
            f"Installing PyTorch into {ADAPTFM_ENV}",
            self._on_install_done,
        )

    def _on_install_done(self, exit_code: int):
        if exit_code == 0:
            self._log(
                f"✓ PyTorch installed into '{ADAPTFM_ENV}' successfully.",
                color=_GREEN, bold=True,
            )
        else:
            self._log(
                f"✗ PyTorch install exited with code {exit_code}.\n"
                f"  You can retry from the command line:\n"
                f"  conda run -n {ADAPTFM_ENV} {_read_saved_cmd()}",
                color=_RED, bold=True,
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def current_cmd(self) -> str:
        """Return the currently *saved* command (not the live text-box value)."""
        return _read_saved_cmd()

    def is_configured(self) -> bool:
        return PYTORCH_CMD_FILE.exists() and bool(PYTORCH_CMD_FILE.read_text().strip())