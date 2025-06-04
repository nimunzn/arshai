from typing import List, Optional
from flashrank import Ranker, RerankRequest
import logging
import os
from arshai.core.interfaces import IReranker, IRerankInput
from arshai.core.interfaces import Document


class FlashRankReranker(IReranker):
    """Reranker implementation using FlashRank."""

    def __init__(
        self,
        model_name: str = "rank-T5-flan",
        device: str = 'cpu',
        top_k: Optional[int] = None,
    ) -> None:
        """
        Initialize FlashRankReranker.

        Args:
            model_name: Name of the reranking model to use
            device: Device to run the model on ('cpu' or 'cuda')
            top_k: Number of top results to return (if None, returns all results)
        """
        try:
            self.model_name = model_name
            self.device = device
            self.top_k = top_k
            self.logger = logging.getLogger(__name__)
            
            self.logger.info(f"Initializing FlashRank reranker with model: {model_name}")
            self.reranker = Ranker(model_name=model_name)
        except ImportError:
            raise ImportError(
                "Could not import flashrank python package. "
                "Please install it with `pip install flashrank`."
            )

    def rerank(self, input: IRerankInput) -> List[Document]:
        """
        Rerank documents using FlashRank.

        Args:
            input: IRerankInput containing query and documents to rerank

        Returns:
            List of reranked Document objects
        """
        try:
            # Create passages with metadata
            passages = [
                {
                    "id": i,
                    "text": doc.page_content,
                    "meta": doc.metadata
                }
                for i, doc in enumerate(input.documents)
            ]
            
            # Create RerankRequest object
            rerank_request = RerankRequest(query=input.query, passages=passages)
            
            # Perform reranking
            rerank_results = self.reranker.rerank(rerank_request)
            
            # Process results and limit to top_k if specified
            ranked_documents = []
            for result in rerank_results:
                doc = Document(
                    page_content=result["text"],
                    metadata={
                        "relevance_score": result["score"],
                        **result["meta"]
                    }
                )
                ranked_documents.append(doc)
            
            if self.top_k:
                ranked_documents = ranked_documents[:self.top_k]
                
            return ranked_documents
            
        except Exception as e:
            self.logger.error(f"Reranking error: {e}")
            return input.documents

    async def arerank(self, input: IRerankInput) -> List[Document]:
        """
        Async version of rerank (currently just calls sync version since FlashRank
        doesn't support async operations natively).

        Args:
            input: IRerankInput containing query and documents to rerank

        Returns:
            List of reranked Document objects
        """
        return self.rerank(input) 