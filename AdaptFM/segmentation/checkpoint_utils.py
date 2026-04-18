import urllib.request



def check_sam2_installed():

    try:
        import sam2

    except ImportError:

        raise RuntimeError(
            "SAM2 is not installed.\n"
            "Install with:\n"
            "pip install AdaptFM[sam2]"
        )



def download_file(url, dest):
    print(f"Downloading {dest.name}...")

    urllib.request.urlretrieve(url, dest)



