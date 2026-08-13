import urllib.request
import sys
import importlib


def check_sam2_installed():

    try:
        import sam2

    except ImportError:

        try:
            if "sam2" in sys.modules:
                importlib.reload(sys.modules["sam2"])
        except:
            raise RuntimeError(
                "SAM2 is not installed.\n"
                "Install with:\n"
                "the environment manager"
            )



def download_file(url, dest):
    print(f"Downloading {dest.name}...")

    urllib.request.urlretrieve(url, dest)



