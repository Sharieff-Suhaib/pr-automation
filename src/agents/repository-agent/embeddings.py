from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class CodeEmbedder:

    def __init__(self):
        print(f"Loading embedding model: {MODEL_NAME}")

        self.model = SentenceTransformer(MODEL_NAME)

        print("Embedding model loaded.")

    def create_text(self, node):
        """
        Convert a parsed code node into searchable text.
        """

        return f"""
File: {node['file']}
Language: {node['language']}
Type: {node['type']}
Name: {node['name']}

Code:
{node['code']}
"""

    def embed_nodes(self, nodes):
        """
        Generate embeddings for a list of code nodes.
        """

        texts = [
            self.create_text(node)
            for node in nodes
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True
        )

        return embeddings

    def embed_query(self, query):
        """
        Generate an embedding for a user issue/query.
        """

        embedding = self.model.encode(
            query,
            convert_to_numpy=True
        )

        return embedding