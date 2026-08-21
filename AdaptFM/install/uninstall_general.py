#!/usr/bin/env python3
"""
Generic uninstaller for AdaptFM conda environments.

Usage:
    adaptfm-uninstall <env_name> 
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"  + {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)

def _get_cloned_paths(prefix_path: Path) -> list[Path]:
    """Inspects the conda activate hooks to find cloned external repositories."""
    cloned_paths = []
    activate_dir = prefix_path / "etc" / "conda" / "activate.d"
    if not activate_dir.exists():
        return cloned_paths

    for hook_file in activate_dir.glob("*.sh"):
        try:
            content = hook_file.read_text()
            for line in content.splitlines():
                if line.startswith("export ") and "_ROOT=" in line:
                    # Split 'export SAMMED3D_ROOT=/path' into variable name and path
                    assignment = line.replace("export ", "").strip()
                    var_name, path_str = assignment.split("=", 1)
                    
                    # CRITICAL SAFETY: Never delete the core AdaptFM repository
                    if var_name.strip() == "ADAPTFM_ROOT":
                        continue
                        
                    path = Path(path_str.strip()).expanduser().resolve()
                    if path.exists() and path not in cloned_paths:
                        cloned_paths.append(path)
        except Exception as e:
            print(f"  Warning: Could not parse hook file {hook_file}: {e}")
            
    return cloned_paths

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uninstall an AdaptFM conda environment and clean up associated files."
    )
    parser.add_argument(
        "env_name",
        type=str,
        help="The name of the conda environment to remove (e.g., sammed3d_adapt).",
    )

    args = parser.parse_args()

    env_name = args.env_name
    config_path = Path.home() / ".adaptfm" / f"{env_name}.prefix"

    print(f"\n=== Uninstalling Environment: {env_name} ===\n")

    # 1. Detect the environment prefix and find cloned repos BEFORE deleting the env
    conda_prefix = None
    cloned_repos = []

    if config_path.exists():
        conda_prefix = Path(config_path.read_text().strip())
        if conda_prefix.exists():
            print(f"-> Found environment prefix at: {conda_prefix}")
            cloned_repos = _get_cloned_paths(conda_prefix)
    else:
        print(f"Warning: Configuration file {config_path} not found.")
        print("Will attempt standard conda environment removal.")

    # 2. Remove the Conda Environment
    env_check = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True,
        text=True,
        check=True,
    )
    
    if env_name in env_check.stdout:
        print(f"\n--- Removing Conda Environment '{env_name}' ---")
        _run(["conda", "env", "remove", "-n", env_name, "-y"])
    else:
        print(f"\nConda environment '{env_name}' does not exist. Skipping.")

    # 3. Clean up the .adaptfm prefix file
    if config_path.exists():
        print("\n--- Removing tracking file ---")
        config_path.unlink()
        print(f"  Removed {config_path}")

    # 4. Safely handle cloned repositories
    if cloned_repos:
        print("\n--- Checking Cloned Repositories ---")
        for repo in cloned_repos:
            # Quick safety guard: Don't accidentally delete the current working directory
            if repo == Path.cwd().resolve():
                print(f"  Skipping {repo} (It is your current working directory).")
                continue

            # Check if it looks like a git repo we cloned
            if (repo / ".git").exists():
                print(f"  Removing repository: {repo}")
                shutil.rmtree(repo)


    print(f"\n✓ Uninstallation of '{env_name}' complete.\n")

if __name__ == "__main__":
    main()