from clone_repo import clone_repository


repo1 = "https://github.com/Sharieff-Suhaib/dummy_repo.git"

# repo2 = "https://github.com/Sharieff-Suhaib/image-captioning.git"


path1 = clone_repository(repo1)
# path2 = clone_repository(repo2)


print("\nRepositories:")
print("Repo 1:", path1)
# print("Repo 2:", path2)