#!/usr/bin/env python3
"""
Script to update all imports from old structure to new arshai package structure.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

# Import mappings
IMPORT_MAPPINGS = [
    # Seedwork interfaces to arshai.core.interfaces
    (r'from seedwork\.interfaces\.(\w+) import', r'from arshai.core.interfaces import'),
    (r'import seedwork\.interfaces\.(\w+)', r'import arshai.core.interfaces'),
    (r'from seedwork\.interfaces import', r'from arshai.core.interfaces import'),
    
    # Src modules to arshai modules
    (r'from src\.agents', r'from arshai.agents'),
    (r'from src\.workflows', r'from arshai.workflows'),
    (r'from src\.memory', r'from arshai.memory'),
    (r'from src\.tools', r'from arshai.tools'),
    (r'from src\.llms', r'from arshai.llms'),
    (r'from src\.embeddings', r'from arshai.embeddings'),
    (r'from src\.document_loaders', r'from arshai.document_loaders'),
    (r'from src\.vector_db', r'from arshai.vector_db'),
    (r'from src\.config', r'from arshai.config'),
    (r'from src\.utils', r'from arshai.utils'),
    (r'from src\.factories', r'from arshai.factories'),
    (r'from src\.callbacks', r'from arshai.callbacks'),
    (r'from src\.prompts', r'from arshai.prompts'),
    (r'from src\.rerankers', r'from arshai.rerankers'),
    (r'from src\.speech', r'from arshai.speech'),
    (r'from src\.web_search', r'from arshai.web_search'),
    (r'from src\.clients', r'from arshai.clients'),
    (r'from src\.indexings', r'from arshai.indexings'),
    
    # Import src to import arshai
    (r'import src\.', r'import arshai.'),
    (r'from arshai import', r'from arshai import'),
]

def update_imports_in_file(file_path: Path, dry_run: bool = False) -> List[str]:
    """Update imports in a single file."""
    changes = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all import mappings
        for old_pattern, new_pattern in IMPORT_MAPPINGS:
            matches = re.findall(old_pattern, content)
            if matches:
                content = re.sub(old_pattern, new_pattern, content)
                changes.append(f"Updated: {old_pattern} -> {new_pattern}")
        
        # Only write if changes were made
        if content != original_content:
            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            return changes
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return []

def find_python_files(root_dir: Path, exclude_dirs: List[str] = None) -> List[Path]:
    """Find all Python files in the directory tree."""
    if exclude_dirs is None:
        exclude_dirs = ['.git', '__pycache__', '.pytest_cache', 'build', 'dist', '.venv', 'venv']
    
    python_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Remove excluded directories from the search
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    return python_files

def update_init_files():
    """Update __init__.py files to properly export from new structure."""
    base_path = Path(__file__).parent.parent
    
    # Update arshai/__init__.py
    init_content = '''"""
Arshai: A powerful agent framework for building conversational AI systems.
"""

from arshai._version import __version__, __version_info__
from arshai.config import Settings

# Core interfaces
from arshai.core.interfaces import (
    IAgent,
    IAgentConfig,
    IAgentInput,
    IAgentOutput,
    IWorkflow,
    IWorkflowState,
    ITool,
    IMemoryManager,
    ILLM,
)

# Main implementations
from arshai.agents.conversation import ConversationAgent
from arshai.workflows.workflow_runner import WorkflowRunner
from arshai.workflows.workflow_config import BaseWorkflowConfig

__all__ = [
    # Version
    "__version__",
    "__version_info__",
    # Config
    "Settings",
    # Interfaces
    "IAgent",
    "IAgentConfig",
    "IAgentInput",
    "IAgentOutput",
    "IWorkflow",
    "IWorkflowState",
    "ITool",
    "IMemoryManager",
    "ILLM",
    # Implementations
    "ConversationAgent",
    "WorkflowRunner",
    "BaseWorkflowConfig",
]
'''
    
    init_file = base_path / "arshai" / "__init__.py"
    with open(init_file, 'w') as f:
        f.write(init_content)
    
    print("✓ Updated arshai/__init__.py")

def main():
    """Main function to update all imports."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update imports to new arshai package structure")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    parser.add_argument("--path", type=str, help="Specific path to update (default: entire project)")
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent
    search_path = Path(args.path) if args.path else base_path
    
    print(f"Updating imports in: {search_path}")
    print(f"Dry run: {args.dry_run}")
    print("-" * 50)
    
    # Find all Python files
    python_files = find_python_files(search_path)
    
    total_files_changed = 0
    total_changes = 0
    
    for py_file in python_files:
        changes = update_imports_in_file(py_file, dry_run=args.dry_run)
        if changes:
            total_files_changed += 1
            total_changes += len(changes)
            print(f"\n{py_file.relative_to(base_path)}:")
            for change in changes:
                print(f"  - {change}")
    
    # Update __init__ files if not dry run
    if not args.dry_run and search_path == base_path:
        update_init_files()
    
    print("\n" + "-" * 50)
    print(f"Summary: {total_files_changed} files changed, {total_changes} imports updated")
    
    if args.dry_run:
        print("\nThis was a dry run. Use without --dry-run to apply changes.")

if __name__ == "__main__":
    main()