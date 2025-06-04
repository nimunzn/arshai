"""
Utility functions for document processors.
"""
import json
from pathlib import Path
from typing import List, Dict

def save_docs_to_json(documents: List[Dict], original_filename: str, output_dir_name: str) -> str:
    """
    Save documents to a JSON file.
    
    Args:
        documents (List[Dict]): List of documents to save
        original_filename (str): Original filename
        output_dir_name (str): Directory to save the JSON file
        
    Returns:
        str: Path to the saved file
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename
    output_filename = Path(original_filename).with_suffix('.json').name
    output_path = output_dir / output_filename
    
    # Save documents
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
        
    return str(output_path) 