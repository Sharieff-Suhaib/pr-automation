import os
from git import Repo


def clone_repository(url, base_path="./repositories"):
    """
    Clone a GitHub repository into its own directory.

    Returns the local path of the cloned repository.
    """

    os.makedirs(base_path, exist_ok=True)

    repo_name = url.rstrip("/").split("/")[-1]

    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    repo_path = os.path.join(base_path, repo_name)

    if os.path.exists(repo_path):
        print(f"Repository already exists: {repo_path}")
        return repo_path

    print(f"Cloning {url}...")
    
    Repo.clone_from(url, repo_path)

    print(f"Repository cloned to: {repo_path}")

    return repo_path