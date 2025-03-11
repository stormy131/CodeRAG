from pathlib import Path

from git import Repo


def fetch_data(repo_url: str, data_dir: Path):
    Repo.clone_from(repo_url, data_dir)


if __name__ == "__main__":
    pass
