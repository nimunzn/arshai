#!/usr/bin/env python3
"""
Comprehensive migration script to move files to new arshai package structure.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List

def create_directories(base_path: Path) -> None:
    """Create the new directory structure."""
    directories = [
        "arshai/agents",
        "arshai/callbacks", 
        "arshai/clients",
        "arshai/clients/arshai",
        "arshai/clients/utils",
        "arshai/config",
        "arshai/core/interfaces",
        "arshai/core/base",
        "arshai/document_loaders",
        "arshai/document_loaders/file_loaders",
        "arshai/document_loaders/processors",
        "arshai/document_loaders/text_splitters",
        "arshai/embeddings",
        "arshai/factories",
        "arshai/indexings",
        "arshai/llms",
        "arshai/memory",
        "arshai/memory/long_term",
        "arshai/memory/short_term", 
        "arshai/memory/working_memory",
        "arshai/prompts",
        "arshai/rerankers",
        "arshai/speech",
        "arshai/tools",
        "arshai/utils",
        "arshai/vector_db",
        "arshai/web_search",
        "arshai/workflows",
        "arshai/extensions",
    ]
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("✓ Created directory structure")

def copy_files(base_path: Path) -> Dict[str, List[str]]:
    """Copy files from old structure to new structure."""
    copy_mappings = {
        # Interfaces from seedwork
        "seedwork/interfaces/*.py": "arshai/core/interfaces/",
        
        # Source modules
        "src/agents/*.py": "arshai/agents/",
        "src/callbacks/*.py": "arshai/callbacks/",
        "src/clients/*.py": "arshai/clients/",
        "src/clients/arshai/*.py": "arshai/clients/arshai/",
        "src/clients/utils/*.py": "arshai/clients/utils/",
        "src/config/*.py": "arshai/config/",
        "src/document_loaders/*.py": "arshai/document_loaders/",
        "src/document_loaders/file_loaders/*.py": "arshai/document_loaders/file_loaders/",
        "src/document_loaders/processors/*.py": "arshai/document_loaders/processors/",
        "src/document_loaders/text_splitters/*.py": "arshai/document_loaders/text_splitters/",
        "src/embeddings/*.py": "arshai/embeddings/",
        "src/factories/*.py": "arshai/factories/",
        "src/indexings/*.py": "arshai/indexings/",
        "src/llms/*.py": "arshai/llms/",
        "src/memory/*.py": "arshai/memory/",
        "src/memory/long_term/*.py": "arshai/memory/long_term/",
        "src/memory/short_term/*.py": "arshai/memory/short_term/",
        "src/memory/working_memory/*.py": "arshai/memory/working_memory/",
        "src/prompts/*.py": "arshai/prompts/",
        "src/rerankers/*.py": "arshai/rerankers/",
        "src/speech/*.py": "arshai/speech/",
        "src/tools/*.py": "arshai/tools/",
        "src/utils/*.py": "arshai/utils/",
        "src/vector_db/*.py": "arshai/vector_db/",
        "src/web_search/*.py": "arshai/web_search/",
        "src/workflows/*.py": "arshai/workflows/",
    }
    
    copied_files = {}
    
    for pattern, dest in copy_mappings.items():
        src_pattern = base_path / pattern
        dest_dir = base_path / dest
        
        # Get all matching files
        if '*' in pattern:
            parent_dir = src_pattern.parent
            if parent_dir.exists():
                files = list(parent_dir.glob(src_pattern.name))
            else:
                files = []
        else:
            files = [src_pattern] if src_pattern.exists() else []
        
        copied_files[pattern] = []
        
        for src_file in files:
            if src_file.is_file():
                dest_file = dest_dir / src_file.name
                shutil.copy2(src_file, dest_file)
                copied_files[pattern].append(src_file.name)
    
    return copied_files

def update_imports_in_file(file_path: Path) -> bool:
    """Update imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Update seedwork imports
        content = content.replace('from seedwork.interfaces', 'from arshai.core.interfaces')
        content = content.replace('import seedwork.interfaces', 'import arshai.core.interfaces')
        
        # Update src imports
        content = content.replace('from src.', 'from arshai.')
        content = content.replace('import arshai.', 'import arshai.')
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def update_all_imports(base_path: Path) -> int:
    """Update imports in all Python files in arshai directory."""
    arshai_path = base_path / "arshai"
    updated_count = 0
    
    for py_file in arshai_path.glob("**/*.py"):
        if update_imports_in_file(py_file):
            updated_count += 1
    
    return updated_count

def update_init_files(base_path: Path) -> None:
    """Create or update __init__.py files."""
    # Main arshai/__init__.py
    init_content = '''"""
Arshai: A powerful agent framework for building conversational AI systems.
"""

from arshai._version import __version__, __version_info__

# Public API - import these last to avoid circular imports
__all__ = [
    "__version__",
    "__version_info__",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "Settings":
        from arshai.config.settings import Settings
        return Settings
    elif name == "IAgent":
        from arshai.core.interfaces import IAgent
        return IAgent
    elif name == "IAgentConfig":
        from arshai.core.interfaces import IAgentConfig
        return IAgentConfig
    elif name == "IAgentInput":
        from arshai.core.interfaces import IAgentInput
        return IAgentInput
    elif name == "ConversationAgent":
        from arshai.agents.conversation import ConversationAgent
        return ConversationAgent
    elif name == "WorkflowRunner":
        from arshai.workflows.workflow_runner import WorkflowRunner
        return WorkflowRunner
    elif name == "BaseWorkflowConfig":
        from arshai.workflows.workflow_config import BaseWorkflowConfig
        return BaseWorkflowConfig
    raise AttributeError(f"module 'arshai' has no attribute '{name}'")
'''
    
    with open(base_path / "arshai" / "__init__.py", 'w') as f:
        f.write(init_content)
    
    print("✓ Updated arshai/__init__.py")

def main():
    """Main migration function."""
    base_path = Path(__file__).parent.parent
    
    print("Starting package structure migration...")
    print(f"Base path: {base_path}")
    print("-" * 50)
    
    # Step 1: Create directories
    create_directories(base_path)
    
    # Step 2: Copy files
    print("\nCopying files...")
    copied_files = copy_files(base_path)
    
    total_files = sum(len(files) for files in copied_files.values())
    print(f"✓ Copied {total_files} files")
    
    # Step 3: Update imports in copied files
    print("\nUpdating imports in copied files...")
    updated_count = update_all_imports(base_path)
    print(f"✓ Updated imports in {updated_count} files")
    
    # Step 4: Update __init__.py files
    print("\nUpdating __init__.py files...")
    update_init_files(base_path)
    
    print("\n" + "-" * 50)
    print("✅ Migration complete!")
    print("\nNext steps:")
    print("1. Run tests to verify everything works")
    print("2. Update remaining imports in tests and examples")
    print("3. Remove old src/ and seedwork/ directories when ready")

if __name__ == "__main__":
    main()