from collections.abc import Callable
from pathlib import Path

from AdaptFM.gui.widgets.sam_card import SamCard

_INSTALLED_COLOR = "#4caf50"
_WARNING_COLOR = "#f44336"


class SSVTCard(SamCard):
    def __init__(
        self,
        *,
        key: str,
        display_name: str,
        description: str,
        import_name: str,
        install_command: str,
        uninstall_command: str,
        requires_pytorch: bool,
        log_fn: Callable[..., None],
        run_process_fn: Callable[..., None],
        parent=None,
        source_dir=None,
    ):
        super().__init__(
            key=key,
            display_name=display_name,
            description=description,
            import_name=import_name,
            install_command=install_command,
            uninstall_command=uninstall_command,
            requires_pytorch=requires_pytorch,
            log_fn=log_fn,
            run_process_fn=run_process_fn,
            parent=parent,
            source_dir=source_dir,
        )

        # Override anything you want AFTER calling super()
        self._description = "Downloads the SSVT model from Hugging Face"
        self._requires_pytorch = False
        self._import_name = None

    def _probe(self) -> bool:

        REPO_ROOT = Path(__file__).parent.parent.parent
        ssvt_dir = REPO_ROOT / "SSVT" / "checkpoint"
        checkpoint = ssvt_dir / "SSVT.pth"
        has_checkpoint = checkpoint.exists()

        return has_checkpoint

    def _on_op_done(self, exit_code: int):
        self.set_busy(False)

        if exit_code == 0:
            self._log("\n✓ Done (exit 0)", color=_INSTALLED_COLOR, bold=True)
            self._installed = self._probe()

        else:
            self._log(
                f"\n✗ Exited with code {exit_code}", color=_WARNING_COLOR, bold=True
            )

        self._pending_op = None
        self._refresh_badge()
