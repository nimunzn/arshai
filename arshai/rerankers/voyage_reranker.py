from typing import List, Optional
import logging
import os
import voyageai
from arshai.core.interfaces.ireranker import IReranker, IRerankInput
from arshai.core.interfaces.idocument import Document

class VoyageReranker(IReranker):
    """Reranker implementation using Voyage AI API."""

    def __init__(
        self,
        model_name: str = 'rerank-2',
        top_k: Optional[int] = None,
        api_key: Optional[str] = None
    ) -> None:
        """
        Initialize VoyageReranker.

        Args:
            model_name: Name of the Voyage reranking model to use (default: rerank-2)
                        Available models: rerank-2 (high quality), rerank-lite-1 (faster)
            top_k: Number of top results to return (if None, returns all results)
            api_key: Voyage AI API key (if None, uses VOYAGE_API_KEY env var)
        """
        self.model_name = model_name
        self.top_k = top_k
        self.logger = logging.getLogger(__name__)
        
        try:
            # Get API key from parameter or environment variable
            api_key = api_key or os.environ.get("VOYAGE_API_KEY")
            if not api_key:
                raise ValueError("Voyage API key not provided. Set it as parameter or VOYAGE_API_KEY environment variable.")
            

            self.logger.info(f"Initializing Voyage reranker with model: {self.model_name}")
            self.client = voyageai.Client(api_key=api_key)
        except ImportError:
            raise ImportError(
                "Could not import voyageai python package. "
                "Please install it with `pip install voyageai`."
            )
        except Exception as e:
            self.logger.error(f"Error initializing Voyage reranker: {str(e)}")
            raise

    def rerank(self, input: IRerankInput) -> List[Document]:
        """
        Rerank documents using Voyage AI API.

        Args:
            input: IRerankInput containing query and documents

        Returns:
            List of reranked Document objects
        """
        try:
            # Extract document contents
            documents_contents = [doc.page_content for doc in input.documents]
            
            # Perform reranking
            reranking = self.client.rerank(
                query=input.query,
                documents=documents_contents,
                model=self.model_name,
                top_k=self.top_k if self.top_k else len(documents_contents)
            )
            
            # Create reranked document list
            ranked_documents = []
            for result in reranking.results:
                original_doc = input.documents[result.index]
                # Update metadata with relevance score
                metadata = {
                    **original_doc.metadata,
                    "relevance_score": result.relevance_score
                }
                ranked_documents.append(Document(
                    page_content=result.document,
                    metadata=metadata
                ))
            
            return ranked_documents
            
        except Exception as e:
            self.logger.error(f"Voyage reranking error: {e}")
            return input.documents

    async def arerank(self, input: IRerankInput) -> List[Document]:
        """
        Async version of rerank (currently just calls sync version since 
        voyageai library doesn't support async operations natively).

        Args:
            input: IRerankInput containing query and documents

        Returns:
            List of reranked Document objects
        """
        return self.rerank(input) 