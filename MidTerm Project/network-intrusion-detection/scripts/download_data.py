from pathlib import Path
import urllib.request

# Always resolve paths relative to the project root
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
TEST_URL  = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

def download_file(url: str, filename: str):
    filepath = DATA_DIR / filename
    if filepath.exists():
        print(f"{filename} already exists → skipping")
        return
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filepath)
    print(f"Saved → {filepath}")

if __name__ == "__main__":
    download_file(TRAIN_URL, "KDDTrain+.txt")
    download_file(TEST_URL, "KDDTest+.txt")
    print("\nDataset ready!")