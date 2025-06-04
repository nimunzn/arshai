from typing import List, Dict, Any, Optional, Callable
import json
import logging

from config.settings import Settings
from arshai.core.interfaces.itext_processor import ITextProcessor
from arshai.core.interfaces.idocument import Document
from arshai.document_loaders.config import ContextEnricherConfig
from arshai.core.interfaces.illm import ILLMInput, LLMInputType, ILLMConfig

logger = logging.getLogger(__name__)

class ContextEnricher(ITextProcessor):
    """Processor for enriching documents with additional context.
    
    This processor adds context from surrounding documents to make
    each document more self-contained and useful.
    """
    
    def __init__(self, config: ContextEnricherConfig, setting: Settings):
        """Initialize the context enricher.
        
        Args:
            config: Configuration for the enricher. If None, uses ContextEnricherConfig defaults.
        """
        self.config = config
        self._custom_processors = {}
        self.setting = setting
        # Initialize LLM client if API key is provided
        self.llm_client = self.setting.create_llm()
    
    def register_processor(self, name: str, processor_func: Callable[[List[Document]], List[Document]]) -> None:
        """Register a custom processor function.
        
        Args:
            name: Name of the processor
            processor_func: Function that takes and returns a list of Documents
        """
        if name in self._custom_processors:
            logger.warning(f"Processor with name {name} already exists and will be overwritten")
            
        self._custom_processors[name] = processor_func
        logger.info(f"Registered custom processor: {name}")
    
    def process(self, documents: List[Document]) -> List[Document]:
        """Process a list of documents to add context.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of processed Document objects with added context
        """
        if not documents:
            logger.warning("No documents provided for processing")
            return []
            
        if len(documents) == 1:
            logger.info("Only one document provided, no context to add")
            return documents
        
        try:
            enriched_docs = []
            window_size = self.config.window_size
            
            for i, doc in enumerate(documents):
                # Get surrounding documents for context
                start_idx = max(0, i - window_size)
                end_idx = min(len(documents) - 1, i + window_size)
                
                surrounding_docs = documents[start_idx:i] + documents[i+1:end_idx+1]
                
                # Enrich document
                enriched_doc = self._enrich_document(doc, surrounding_docs)
                enriched_docs.append(enriched_doc)
            
            return enriched_docs
            
        except Exception as e:
            logger.error(f"Error enriching documents with context: {str(e)}")
            # Return original documents in case of error
            return documents
    
    def process_text(self, text: str) -> Optional[str]:
        """Process a single text string.
        
        This is a minimal implementation since context enricher works on document lists,
        not individual texts.
        
        Args:
            text: Text to process
            
        Returns:
            The original text (context enricher doesn't modify individual texts)
        """
        return text
    
    def _enrich_document(self, doc: Document, surrounding_docs: List[Document]) -> Document:
        """Enrich a single document with context from surrounding documents.
        
        Args:
            doc: Document to enrich
            surrounding_docs: List of surrounding documents
            
        Returns:
            Enriched Document object
        """
        # Copy metadata to avoid modifying the original
        new_metadata = doc.metadata.copy()
        
        # Add context information
        new_metadata["has_context"] = True
        new_metadata["context_window_size"] = self.config.window_size
        
        # Add surrounding document content (for search or analysis)
        context_texts = [d.page_content for d in surrounding_docs]
        context_combined = "\n".join(context_texts)
        
        # Add surrounding document metadata
        if self.config.include_metadata:
            surrounding_metadata = [d.metadata for d in surrounding_docs]
            new_metadata["surrounding_metadata"] = surrounding_metadata
        
        # Add source documents to metadata if requested
        if self.config.add_source_documents:
            new_metadata["source_documents"] = [
                {
                    "content": d.page_content,
                    "metadata": d.metadata
                }
                for d in surrounding_docs
            ]
        
        # For now, we don't modify the content, just the metadata
        # But we could add context directly to the content if needed
        
        return Document(
            page_content=doc.page_content,
            metadata=new_metadata
        )
    
    def add_wider_window(self, documents: List[Document]) -> List[Document]:
        """Add wider context window to each document by including text from adjacent chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of Document objects with wider window
        """
        if not documents:
            logger.warning("No documents provided for adding wider window")
            return []
            
        if len(documents) == 1:
            logger.info("Only one document provided, no wider window to add")
            return documents
            
        processed_docs = []
        for i, doc in enumerate(documents):
            wider_context = []
            
            # Add previous chunk if available
            if i > 0:
                wider_context.append(documents[i-1].page_content)
                
            # Add current chunk
            wider_context.append(doc.page_content)
            
            # Add next chunk if available
            if i < len(documents) - 1:
                wider_context.append(documents[i+1].page_content)
                
            # Create new metadata with wider window
            metadata = doc.metadata.copy()
            metadata["wider_window"] = "\n\n".join(wider_context)
            
            processed_docs.append(Document(
                page_content=doc.page_content,
                metadata=metadata
            ))
            
        return processed_docs
    
    def add_gpt_context(self, documents: List[Document]) -> List[Document]:
        """Add context to document chunks using GPT.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of Document objects with added context
        """
        if not self.config.api_key:
            logger.error("API key is required for GPT context enrichment")
            raise ValueError("API key is required for GPT context enrichment")
            
        # Initialize LLM client if not already initialized
        if not self.llm_client:
            self.llm_client = self.setting.create_llm()
        
        system_prompt = """
        You are a document context analyzer. Your task is to:
        1. Analyze the given text chunk
        2. Identify its main topics and themes
        3. Provide a brief summary (2-3 sentences)
        4. List 3-5 key concepts or terms
        
        Format your response as JSON with these fields:
        {
            "summary": "Brief summary here",
            "key_concepts": ["concept1", "concept2", "concept3"],
            "topics": ["topic1", "topic2"]
        }
        """
        
        processed_docs = []
        for doc in documents:
            try:
                # Prepare input for the LLM client
                llm_input = ILLMInput(
                    input_type=LLMInputType.CHAT_COMPLETION,
                    system_prompt=system_prompt,
                    user_message=doc.page_content
                )
                
                # Send to the LLM
                response = self.llm_client.chat_completion(llm_input)
                
                # Extract and parse the response
                if isinstance(response, dict) and "llm_response" in response:
                    response_content = response["llm_response"]
                    context = json.loads(response_content) if isinstance(response_content, str) else response_content
                    
                    metadata = doc.metadata.copy()
                    metadata["context"] = context
                    
                    processed_docs.append(Document(
                        page_content=doc.page_content,
                        metadata=metadata
                    ))
                else:
                    # If response format is unexpected, keep the original document
                    logger.warning("Unexpected response format from LLM")
                    processed_docs.append(doc)
                
            except Exception as e:
                logger.error(f"Error generating GPT context: {str(e)}")
                # If context generation fails, keep the original document
                processed_docs.append(doc)
                
        return processed_docs
    
    def run_custom_processor(self, name: str, documents: List[Document]) -> List[Document]:
        """Run a registered custom processor.
        
        Args:
            name: Name of the processor to run
            documents: List of Document objects
            
        Returns:
            Processed Document objects
        """
        if name not in self._custom_processors:
            error_msg = f"No processor registered with name: {name}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        try:
            return self._custom_processors[name](documents)
        except Exception as e:
            logger.error(f"Error running custom processor {name}: {str(e)}")
            # Return original documents in case of error
            return documents 