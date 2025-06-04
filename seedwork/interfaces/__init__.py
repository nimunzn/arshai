"""
Common interfaces for the seedwork package.
"""

from .idocument import Document
from .itext_splitter import ITextSplitter, ITextSplitterConfig 
from .itext_processor import ITextProcessor, ITextProcessorConfig
from .ifile_loader import IFileLoader, IFileLoaderConfig
from .iembedding import IEmbedding
from .iagent import IAgent
from .idatabase_client import IDatabaseClient
from .iindexing import IIndexing
