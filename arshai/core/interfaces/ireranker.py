from typing import List, Protocol, Optional, Dict, Any
from pydantic import BaseModel, Field
from .idocument import Document
from .idto import IDTO


class IRerankInput(IDTO):
    """
    Represents the input for reranking.
    
    Attributes:
        query: The search query
        documents: List of documents to rerank
        config: Optional configuration for the reranking process
    """
    query: str = Field(description="The search query to use for reranking")
    documents: List[Document] = Field(description="List of documents to be reranked")



class IReranker(Protocol):
    """
    Interface defining the contract for rerankers.
    Any reranker implementation must conform to this interface.
    """
    
    def rerank(self, input: IRerankInput) -> List[Document]:
        """
        Rerank the documents based on their relevance to the query.

        Args:
            input: IRerankInput containing query, documents and optional config

        Returns:
            List of reranked Document objects
        """
        ...

    async def arerank(self, input: IRerankInput) -> List[Document]:
        """
        Asynchronously rerank the documents based on their relevance to the query.

        Args:
            input: IRerankInput containing query, documents and optional config

        Returns:
            List of reranked Document objects
        """
        ... 