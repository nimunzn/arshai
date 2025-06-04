import os
import logging
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
from pdf2image import convert_from_path
from tqdm import tqdm

from arshai.core.interfaces import IIndexing
from arshai.config.settings import Settings

logger = logging.getLogger(__name__)

class MultimodalIndexer(IIndexing):
    """
    Indexer for multimodal documents (PDFs with images) using VoyageAI embeddings and Milvus.
    """
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the multimodal indexer with settings.
        
        Args:
            settings: Application settings
        """
        self.settings = settings or Settings()
        
        # Use Settings to create vector_db, collection_config, and embedding_model
        self.vector_db, self.collection_config, self.embedding_model = self.settings.create_vector_db()
        
        # Get documents directory
        self.documents_directory = self.settings.get("documents_directory", "documents")
        
        # Create images directory if it doesn't exist
        Path("pages").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized MultimodalIndexer with collection: {self.collection_config.collection_name}")

    def _convert_pdf_to_images(self, pdf_path: str, save_pages: bool = False, pages_dir: str = "pages") -> Dict[str, Image.Image]:
        """
        Convert a PDF to images.
        
        Args:
            pdf_path: Path to the PDF file
            save_pages: Whether to save the page images to disk
            pages_dir: Directory to save page images
            
        Returns:
            Dictionary mapping page names to PIL Images
        """
        logger.info(f"Converting PDF to images: {pdf_path}")
        try:
            images = convert_from_path(pdf_path)
            images_dictionary = {}
            
            # Ensure pages directory exists
            Path(pages_dir).mkdir(parents=True, exist_ok=True)
            
            for i, image in enumerate(images):
                page_name = f"page_{i + 1}"
                images_dictionary[page_name] = image
                
                # Save page as PNG if requested
                if save_pages:
                    page_path = os.path.join(pages_dir, f"{page_name}.png")
                    logger.info(f"Saving page image to: {page_path}")
                    image.save(page_path, "PNG")
            
            return images_dictionary
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert a PIL Image to a base64 encoded string.
        
        Args:
            image: PIL Image to convert
            
        Returns:
            Base64 encoded string representation of the image
        """

        prefix = "data:image/jpeg;base64,"

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (max dimension 1024)
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.LANCZOS)
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Check if base64 string is within size limit
        max_chars = 65535
        if len(img_str) > max_chars:
            logger.info(f"Image is too large, reducing quality and size")
            # Try with progressively lower quality until it fits
            quality = 80
            while len(img_str) > max_chars and quality >= 50:
                buffered = BytesIO()
                image.save(buffered, format="JPEG", quality=quality)
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                quality -= 5
            
            # If still too large, reduce dimensions further
            scale = 0.9
            while len(img_str) > max_chars and scale > 0.3:
                buffered = BytesIO()
                new_size = tuple(int(dim * scale) for dim in image.size)
                smaller_image = image.resize(new_size, Image.LANCZOS)
                smaller_image.save(buffered, format="JPEG", quality=quality)
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                scale -= 0.1
        
        logger.debug(f"Image encoded to base64: {len(img_str)} characters, dimensions: {image.size}")
        return prefix + img_str
        
    def _process_base64_image(self, base64_data: str) -> str:
        """
        Process a base64 image to ensure it meets size requirements.
        
        Args:
            base64_data: Base64 encoded image
            
        Returns:
            Processed base64 encoded image
        """
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(base64_data)
            image = Image.open(BytesIO(image_bytes))
            
            # Use the _image_to_base64 method to process and re-encode
            return self._image_to_base64(image)
        except Exception as e:
            logger.error(f"Error processing base64 image: {str(e)}")
            return base64_data  # Return original if processing fails
    
    def create_collection(self) -> None:
        """
        Create the collection for multimodal embeddings.
        This is handled by vector_db.get_or_create_collection()
        """
        logger.info(f"Creating collection: {self.collection_config.collection_name}")
        try:
            # Connect to Vector DB
            self.vector_db.connect()
            
            # Let the vector_db client handle collection creation
            collection = self.vector_db.get_or_create_collection(self.collection_config)
            
            logger.info(f"Collection created: {self.collection_config.collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise
    
    def index_document(self, pdf_path: str, save_pages: bool = False, pages_dir: str = "pages") -> None:
        """
        Index a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            save_pages: Whether to save the page images to disk
            pages_dir: Directory to save page images
        """
        logger.info(f"Indexing PDF: {pdf_path}")
        try:
            # Convert PDF to images
            images_dict = self._convert_pdf_to_images(pdf_path, save_pages, pages_dir)
            
            # Process each page
            for i, (page_name, image) in enumerate(tqdm(images_dict.items(), desc="Embedding pages")):
                try:
                    # Generate embeddings using the embedding model from settings
                    embedding_result = self.embedding_model.multimodel_embed(input=[image])
                    
                    # Convert image to base64 string
                    image_base64 = self._image_to_base64(image)
                    
                    # Create metadata
                    metadata = {
                        "source": os.path.basename(pdf_path),
                        "page": page_name,
                        "page_number": i + 1,
                        "total_pages": len(images_dict),
                    }
                    
                    # Create entity data with image base64 as text_field
                    entity = {
                        self.collection_config.text_field: image_base64,
                        self.collection_config.metadata_field: metadata
                    }
                    
                    # Create document embedding dictionary
                    documents_embedding = {
                        "dense": embedding_result
                    }
                    
                    # Debug logs for field names
                    logger.info(f"Collection config: text_field={self.collection_config.text_field}, "
                               f"metadata_field={self.collection_config.metadata_field}, "
                               f"dense_field={self.collection_config.dense_field}")
                    logger.info(f"Entity keys: {entity.keys()}")
                    logger.info(f"Documents embedding keys: {documents_embedding.keys()}")
                    
                    # Insert into vector DB using the vector_db instance from settings
                    self.vector_db.insert_entity(
                        config=self.collection_config,
                        entity=entity,
                        documents_embedding=documents_embedding
                    )
                    
                    logger.debug(f"Indexed {page_name} from {pdf_path}")
                except Exception as e:
                    logger.error(f"Error processing page {page_name}: {str(e)}")
            
            logger.info(f"Successfully indexed PDF: {pdf_path}")
        except Exception as e:
            logger.error(f"Error indexing PDF: {str(e)}")
            raise
    
    def index_directory(self, save_pages: bool = False, pages_dir: str = "pages") -> None:
        """
        Index all PDF documents in the configured directory.
        
        Args:
            save_pages: Whether to save the page images to disk
            pages_dir: Directory to save page images
        """
        logger.info(f"Indexing all PDFs in directory: {self.documents_directory}")
        try:
            # Ensure collection exists
            self.create_collection()
            
            # Find all PDFs
            pdf_files = [f for f in os.listdir(self.documents_directory) if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                logger.warning(f"No PDF files found in {self.documents_directory}")
                return
            
            logger.info(f"Found {len(pdf_files)} PDF files to index")
            
            # Process each PDF
            for pdf_file in tqdm(pdf_files, desc="Indexing PDFs"):
                pdf_path = os.path.join(self.documents_directory, pdf_file)
                self.index_pdf(pdf_path, save_pages, pages_dir)
            
            logger.info(f"Successfully indexed {len(pdf_files)} PDF files")
        except Exception as e:
            logger.error(f"Error indexing directory: {str(e)}")
            raise
      