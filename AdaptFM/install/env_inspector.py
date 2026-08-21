"""
env_inspector.py
----------------
Pure-Python helpers that query the system (conda, pip, PyPI, GitHub) to
determine install state and available updates.  All network / subprocess
calls are designed to be run from a QThread worker, never from the Qt
main thread.
"""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass

from AdaptFM.install.env_registry import EnvironmentSpec, PipPackageSpec

# ---------------------------------------------------------------------------
# Data classes returned to the GUI
# ---------------------------------------------------------------------------


@dataclass
class PackageVersionInfo:
    """Version state for one pip package inside a conda env."""

    spec: PipPackageSpec
    installed_version: str | None  # None → not installed
    latest_version: str | None  # None → could not determine
    update_available: bool = False
    error: str = ""


@dataclass
class EnvStatus:
    """Full status snapshot for one EnvironmentSpec."""

    spec: EnvironmentSpec
    is_installed: bool
    packages: list[PackageVersionInfo]
    error: str = ""


# ---------------------------------------------------------------------------
# Conda helpers
# ---------------------------------------------------------------------------


def _conda_envs() -> set[str]:
    """Return the set of currently-existing conda env names."""
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        data = json.loads(result.stdout)
        # each path ends with the env name
        return {
            p.rstrip("/\\").split("/")[-1].split("\\")[-1] for p in data.get("envs", [])
        }
    except Exception:
        # fall back to text parsing
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        names: set[str] = set()
        for line in result.stdout.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                names.add(line.split()[0])
        return names


def _conda_run_pip_show(env_name: str, package: str) -> str | None:
    """Return installed version string for *package* in *env_name*, or None."""
    try:
        result = subprocess.run(
            [
                "conda",
                "run",
                "-n",
                env_name,
                "--no-capture-output",
                "pip",
                "show",
                package,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        for line in result.stdout.splitlines():
            if line.lower().startswith("version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Remote version helpers
# ---------------------------------------------------------------------------


def _pypi_latest(package: str) -> str | None:
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            return data["info"]["version"]
    except Exception:
        return None


def _github_latest_tag(repo: str) -> str | None:
    """Return the tag_name of the latest GitHub release (no auth needed for public repos)."""
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("tag_name", "").lstrip("v")
    except Exception:
        return None


def _conda_latest(conda_package: str) -> str | None:
    try:
        result = subprocess.run(
            ["conda", "search", conda_package, "--json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        data = json.loads(result.stdout)
        entries = data.get(conda_package, [])
        if entries:
            return entries[-1].get("version")
    except Exception:
        pass
    return None


def _version_tuple(v: str):
    """Convert "1.2.3" → (1, 2, 3) for simple comparisons."""
    try:
        return tuple(int(x) for x in v.lstrip("v").split(".")[:3])
    except Exception:
        return (0,)


def _update_available(installed: str | None, latest: str | None) -> bool:
    if not installed or not latest:
        return False
    return _version_tuple(latest) > _version_tuple(installed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def probe_env(spec: EnvironmentSpec) -> EnvStatus:
    """
    Synchronously probe one environment.  Call from a background thread.
    Returns an EnvStatus populated with install state and version info.
    """
    existing_envs = _conda_envs()
    is_installed = spec.conda_env_name in existing_envs

    pkg_infos: list[PackageVersionInfo] = []
    if is_installed:
        for pkg_spec in spec.pip_packages:
            installed_ver = _conda_run_pip_show(
                spec.conda_env_name, pkg_spec.import_name
            )

            latest_ver: str | None = None
            error = ""
            try:
                if pkg_spec.update_source == "pypi":
                    latest_ver = _pypi_latest(pkg_spec.import_name)
                elif pkg_spec.update_source == "github" and pkg_spec.github_repo:
                    latest_ver = _github_latest_tag(pkg_spec.github_repo)
                elif pkg_spec.update_source == "conda":
                    name = pkg_spec.conda_package or pkg_spec.import_name
                    latest_ver = _conda_latest(name)
                # source == "none" → leave latest_ver as None
            except Exception as exc:
                error = str(exc)

            pkg_infos.append(
                PackageVersionInfo(
                    spec=pkg_spec,
                    installed_version=installed_ver,
                    latest_version=latest_ver,
                    update_available=_update_available(installed_ver, latest_ver),
                    error=error,
                )
            )

    return EnvStatus(
        spec=spec,
        is_installed=is_installed,
        packages=pkg_infos,
    )


def probe_all(specs: list[EnvironmentSpec]) -> list[EnvStatus]:
    """Probe every environment in *specs* sequentially."""
    return [probe_env(s) for s in specs]
