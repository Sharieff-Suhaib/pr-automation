from clone_repo import clone_repository
from code_parser import get_source_files, extract_code_nodes


repo_url = "https://github.com/Sharieff-Suhaib/dummy_repo.git"

repo_path = clone_repository(repo_url)

files = get_source_files(repo_path)

print("\nSOURCE FILES")
print("=" * 50)

for file in files:
    print(file)


print("\nCODE NODES")
print("=" * 50)

for file in files:

    nodes = extract_code_nodes(file, repo_path)

    for node in nodes:

        print("\nFile:", node["file"])
        print("Language:", node["language"])
        print("Type:", node["type"])
        print("Name:", node["name"])
        print("Parent:", node["parent"])
        print(
            "Lines:",
            node["start_line"],
            "-",
            node["end_line"]
        )
        print("Code:")
        print(node["code"])