import asyncio
from functools import partial
from typing import Dict, List, Iterable, Tuple, Optional, Any

from attr import define
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document as LangchainDocument
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import VectorStore, SKLearnVectorStore

default_openai_embedding_model_name = "text-embedding-ada-002"


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

    class AsyncSKLearnVectorStore(SKLearnVectorStore):
        """Async version of SKLearnVectorStore."""

        async def aadd_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
        ) -> List[str]:
            func = partial(self.add_texts, texts, metadatas, **kwargs)
            return await asyncio.get_event_loop().run_in_executor(None, func)

    def __init__(self, embeddings: Embeddings = None, storage: VectorStore = None):
        """Initialize the agent."""

        # Use OpenAI by default.
        self._embeddings_model = (
            embeddings if embeddings is not None else OpenAIEmbeddings()
        )

        # Use sklearn (brute-force) by default.
        self._storage = (
            storage
            if storage is not None
            else self.AsyncSKLearnVectorStore(self._embeddings_model, algorithm="brute")
        )

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
