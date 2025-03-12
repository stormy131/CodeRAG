from pathlib import Path

from git import Repo


GIT_REPO = "https://github.com/viarotel-org/escrcpy.git"

# TODO: Path config
def main():
    Repo.clone_from(GIT_REPO, Path("../data"))


if __name__ == "__main__":
    main()
