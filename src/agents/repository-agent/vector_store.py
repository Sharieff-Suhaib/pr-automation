import faiss
import numpy as np
import pickle


class VectorStore:

    def __init__(self, dimension):
        """
        Create a FAISS index.

        dimension:
            Size of each embedding vector.
            all-MiniLM-L6-v2 = 384
        """

        self.dimension = dimension

        self.index = faiss.IndexFlatL2(dimension)

        self.metadata = []

    def add(self, embeddings, nodes):
        """
        Add embeddings and their corresponding metadata.
        """

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        self.index.add(embeddings)

        self.metadata.extend(nodes)

    def search(self, query_embedding, top_k=5):
        """
        Search for the most similar code nodes.
        """

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        query_embedding = query_embedding.reshape(
            1,
            -1
        )

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for distance, index in zip(
            distances[0],
            indices[0]
        ):

            if index == -1:
                continue

            result = self.metadata[index].copy()

            result["distance"] = float(distance)

            results.append(result)

        return results

    def save(self, index_path, metadata_path):
        """
        Save FAISS index and metadata.
        """

        faiss.write_index(
            self.index,
            index_path
        )

        with open(metadata_path, "wb") as f:
            pickle.dump(
                self.metadata,
                f
            )

    def load(self, index_path, metadata_path):
        """
        Load FAISS index and metadata.
        """

        self.index = faiss.read_index(
            index_path
        )

        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)