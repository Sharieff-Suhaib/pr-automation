from clone_repo import clone_repository
from code_parser import get_source_files, extract_code_nodes
from embeddings import CodeEmbedder
from vector_store import VectorStore


repo_url = "https://github.com/Sharieff-Suhaib/dummy_repo.git"

repo_path = clone_repository(repo_url)

files = get_source_files(repo_path)

print("\nSOURCE FILES")

for file in files:
    print(file)


nodes = []

for file in files:

    file_nodes = extract_code_nodes(
        file,
        repo_path
    )

    nodes.extend(file_nodes)


print("\nTOTAL PARSED NODES:", len(nodes))


# Remove class nodes from embedding
embedding_nodes = [
    node
    for node in nodes
    if node["type"] != "class"
]


print("NODES USED FOR EMBEDDING:", len(embedding_nodes))


print("\nNODES USED FOR EMBEDDING")

for node in embedding_nodes:

    print(
        f"{node['file']} → "
        f"{node['type']} → "
        f"{node['name']}"
    )


embedder = CodeEmbedder()

embeddings = embedder.embed_nodes(
    embedding_nodes
)

print(
    "\nEmbedding shape:",
    embeddings.shape
)


dimension = embeddings.shape[1]

store = VectorStore(dimension)


store.add(
    embeddings,
    embedding_nodes
)

print(
    "FAISS index size:",
    store.index.ntotal
)


query = "Login fails when username is empty"

query_embedding = embedder.embed_query(
    query
)


results = store.search(
    query_embedding,
    top_k=5
)


print("\nSEARCH RESULTS")

for i, result in enumerate(results,start=1):

    print(f"\n{i}.")
    print("File:",result["file"])
    print("Language:",result["language"])
    print("Type:",result["type"])
    print("Name:",result["name"])
    print("Parent:",result["parent"])
    print("Distance:",result["distance"])

    print("Code:")

    print(result["code"])