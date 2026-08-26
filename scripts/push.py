import os
from git import Repo
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, "../.env")

load_dotenv(dotenv_path)

local_dir = "./my_cloned_repo"
username = "Sharieff-Suhaib"
repo_name = "dummy_repo" 

github_token = os.getenv("GITHUB_TOKEN")

if not github_token:
    raise ValueError("GITHUB_TOKEN environment variable not found. Please check your .env path.")

auth_url = f"https://{github_token}@github.com/{username}/{repo_name}.git"

repo = Repo(local_dir)

origin = repo.remotes.origin
origin.set_url(auth_url)

print("Pushing commits to GitHub...")

push_result = origin.push()

for info in push_result:
    if info.flags & info.ERROR:
        print(f"Error pushing: {info.summary}")
    else:
        print(f"Success! {info.summary}")