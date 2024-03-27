import uuid
from typing import Dict, List, Iterable, Tuple, Optional, Any, Callable

import numpy as np
from attr import define
from langchain.schema import Document as LangchainDocument
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import VectorStore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings

default_openai_embedding_model_name = "text-embedding-ada-002"


class SimpleVectorStore(VectorStore):
    """Simple vector store for testing."""

    def __init__(self, embedding_model: Embeddings):
        """Initialize the vector store."""
        self.embedding_model = embedding_model
        self.vectors: List[np.ndarray] = []
        self.ids: List[str] = []
        self.docstore = InMemoryDocstore()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store."""
        embedding_values = self.embedding_model.embed_documents(list(texts))
        if ids is None:
            ids = [uuid.uuid4().hex for _ in texts]
        self.ids.extend(ids)
        self.vectors.extend(np.array(embedding_values))
        if metadatas is not None:
            docs = [
                LangchainDocument(page_content=text, metadata=metadata)
                for text, metadata in zip(texts, metadatas)
            ]
        else:
            docs = [LangchainDocument(text) for text in texts]

        self.docstore.add(dict(zip(ids, docs)))
        return ids

    def similarity_search(
        self, query: str, k: int = 5, **kwargs
    ) -> List[LangchainDocument]:
        """Find similar vectors."""
        results = []
        for doc, score in self.similarity_search_with_score(query, k, **kwargs):
            results.append(doc)
        return results

    def similarity_search_with_score(
        self, query: str, k: int = 5, **kwargs
    ) -> List[Tuple[LangchainDocument, float]]:
        """Find similar vectors with relevance scores."""

        query_embedding = self.embedding_model.embed_query(query)
        cos_sim = np.array(
            [
                self.cosine_similarity_numpy(np.array(query_embedding, ndmin=2), vector)
                for vector in self.vectors
            ]
        )

        # Exact match is 1, opposite is -1 for cosine similarity, so normalize to 0-1.
        scores = (cos_sim.flatten() + 1) / 2

        # argsort returns the indices that would sort the array, so we reverse it to get the highest scores first.
        indices = np.argsort(
            scores,
            axis=0,
        )[
            ::-1
        ][:k]
        results = []
        for i in indices:
            doc = self.docstore.search(self.ids[i])
            assert isinstance(doc, LangchainDocument)
            results.append((doc, scores[i]))
        return results

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "SimpleVectorStore":
        """Create a vector store from texts."""
        obj = cls(embedding_model=embedding)
        obj.add_texts(texts, metadatas=metadatas)
        return obj

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Select a relevance score function."""
        return lambda x: x

    @staticmethod
    def cosine_similarity_numpy(x: np.ndarray, y: np.ndarray) -> float:

        # Compute the dot product between x and y
        dot_product = np.dot(x, y)

        # Compute the L2 norms (magnitudes) of x and y
        magnitude_x = np.sqrt(np.sum(x**2))
        magnitude_y = np.sqrt(np.sum(y**2))

        # Compute the cosine similarity
        cosine_similarity = dot_product / (magnitude_x * magnitude_y)

        return cosine_similarity

    def cosine_similarity_pure(self, v1: List[float], v2: List[float]) -> float:
        import math

        "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
        sumxx, sumxy, sumyy = 0.0, 0.0, 0.0
        for i in range(len(v1)):
            x = v1[i]
            y = v2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        return sumxy / math.sqrt(sumxx * sumyy)


@define
class Document:
    """Class for storing a piece of text and associated metadata. Keeps an abstraction with Langchain's Document class."""

    text: str
    """String text."""
    metadata: dict = {}
    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """

    @staticmethod
    def from_langchain(document: LangchainDocument) -> "Document":
        """Convert a langchain document to a fable document."""
        return Document(text=document.page_content, metadata=document.metadata)

    def to_langchain(self) -> LangchainDocument:
        """Convert a fable document to a langchain document."""
        return LangchainDocument(page_content=self.text, metadata=self.metadata)


class EmbeddingAgent:
    """Does embedding related things like generation, storage, and retrieval."""

    # class FixedUSearch(USearch):
    #
    #     def add_texts(
    #         self,
    #         texts: Iterable[str],
    #         metadatas: Optional[List[Dict]] = None,
    #         ids: Optional[np.ndarray] = None,
    #         **kwargs: Any,
    #     ) -> List[str]:
    #         """Run more texts through the embeddings and add to the vectorstore.
    #
    #         Args:
    #             texts: Iterable of strings to add to the vectorstore.
    #             metadatas: Optional list of metadatas associated with the texts.
    #             ids: Optional list of unique IDs.
    #
    #         Returns:
    #             List of ids from adding the texts into the vectorstore.
    #         """
    #         if not isinstance(self.docstore, AddableMixin):
    #             raise ValueError(
    #                 "If trying to add texts, the underlying docstore should support "
    #                 f"adding items, which {self.docstore} does not"
    #             )
    #
    #         embeddings = self.embedding.embed_documents(list(texts))
    #         documents = []
    #         for i, text in enumerate(texts):
    #             metadata = metadatas[i] if metadatas else {}
    #             documents.append(Document(page_content=text, metadata=metadata))
    #         last_id = int(self.ids[-1]) + 1
    #         if ids is None:
    #             ids = np.array([str(last_id + id) for id, _ in enumerate(texts)])
    #
    #         self.index.add(np.array(ids), np.array(embeddings))
    #         self.docstore.add(dict(zip(ids, documents)))
    #         self.ids.extend(ids)
    #         return ids.tolist()

    # class AsyncSKLearnVectorStore(SKLearnVectorStore):
    #     """Async version of SKLearnVectorStore."""
    #
    #     async def aadd_texts(
    #         self,
    #         texts: Iterable[str],
    #         metadatas: Optional[List[dict]] = None,
    #         **kwargs: Any,
    #     ) -> List[str]:
    #         func = partial(self.add_texts, texts, metadatas, **kwargs)
    #         return await asyncio.get_event_loop().run_in_executor(None, func)
    #
    #     def _select_relevance_score_fn(self) -> Callable[[float], float]:
    #         """Select a relevance score function."""
    #         return lambda x: 0.5

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        storage: Optional[VectorStore] = None,
    ):
        """Initialize the agent."""

        # Use OpenAI by default.
        self._embeddings_model = (
            embeddings if embeddings is not None else OpenAIEmbeddings()
        )

        # Use sklearn (brute-force) by default.
        if storage is not None:
            self._storage = storage
        else:
            self._storage = SimpleVectorStore(embedding_model=self._embeddings_model)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a text."""
        return await self._embeddings_model.aembed_documents(texts)

    async def embed_query(self, text: str) -> List[float]:
        """Embed a query."""
        return await self._embeddings_model.aembed_query(text)

    async def store_documents(self, docs: List[Document]) -> List[str]:
        """Store a document."""
        return await self._storage.aadd_documents([d.to_langchain() for d in docs])

    async def find_similar(
        self, query: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Find similar documents."""
        results = await self._storage.asimilarity_search_with_relevance_scores(query, k)
        return [(Document.from_langchain(doc), score) for doc, score in results]
