import urllib.request


def check_sam2_installed():
    try:
        import sam2

        return True
    except ImportError:
        raise RuntimeError(
            "SAM2 is not installed.\nYou must install with\nthe model installer"
        )


def check_sam3_installed():
    try:
        import sam3

        return True
    except ImportError:
        raise RuntimeError(
            "SAM3 is not installed.\nYou must install with\nthe model installer"
        )


def download_file(url, dest):
    print(f"Downloading {dest.name}...")

    urllib.request.urlretrieve(url, dest)
