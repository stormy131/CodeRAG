from pathlib import Path

from utils.data import fetch_data


GIT_REPO = "https://github.com/viarotel-org/escrcpy.git"

# TODO: Path config
def main():
    fetch_data(GIT_REPO, Path("../data"))


if __name__ == "__main__":
    main()
