#!/usr/bin/env python3
"""
Script to refactor the package structure for public distribution.
This will move files from the old structure (seedwork/, src/) to the new structure (arshai/).
"""

import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the new directory structure."""
    base_path = Path(__file__).parent.parent
    
    directories = [
        "arshai/core/interfaces",
        "arshai/core/base",
        "arshai/agents",
        "arshai/workflows",
        "arshai/memory",
        "arshai/tools",
        "arshai/llms",
        "arshai/embeddings",
        "arshai/document_loaders",
        "arshai/vector_db",
        "arshai/config",
        "arshai/utils",
        "arshai/extensions",
    ]
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        # Create __init__.py files
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Package initialization."""\n')
    
    print("✓ Created directory structure")
    return base_path

def move_interfaces(base_path):
    """Move interface files to the new structure."""
    src_interfaces = base_path / "seedwork" / "interfaces"
    dst_interfaces = base_path / "arshai" / "core" / "interfaces"
    
    if src_interfaces.exists():
        for file in src_interfaces.glob("*.py"):
            if file.name != "__init__.py":
                shutil.copy2(file, dst_interfaces / file.name)
        print("✓ Moved interface files")

def move_source_files(base_path):
    """Move source files to the new structure."""
    mappings = [
        ("src/agents", "arshai/agents"),
        ("src/workflows", "arshai/workflows"),
        ("src/memory", "arshai/memory"),
        ("src/tools", "arshai/tools"),
        ("src/llms", "arshai/llms"),
        ("src/embeddings", "arshai/embeddings"),
        ("src/document_loaders", "arshai/document_loaders"),
        ("src/vector_db", "arshai/vector_db"),
        ("src/config", "arshai/config"),
        ("src/utils", "arshai/utils"),
        ("src/factories", "arshai/factories"),
        ("src/callbacks", "arshai/callbacks"),
        ("src/prompts", "arshai/prompts"),
        ("src/rerankers", "arshai/rerankers"),
        ("src/speech", "arshai/speech"),
        ("src/web_search", "arshai/web_search"),
        ("src/clients", "arshai/clients"),
    ]
    
    for src, dst in mappings:
        src_path = base_path / src
        dst_path = base_path / dst
        
        if src_path.exists():
            if not dst_path.exists():
                dst_path.mkdir(parents=True, exist_ok=True)
            
            for file in src_path.glob("**/*.py"):
                relative_path = file.relative_to(src_path)
                dst_file = dst_path / relative_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, dst_file)
    
    print("✓ Moved source files")

def update_imports_in_file(file_path):
    """Update imports in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update seedwork imports
    content = content.replace('from seedwork.interfaces', 'from arshai.core.interfaces')
    content = content.replace('import seedwork.interfaces', 'import arshai.core.interfaces')
    
    # Update src imports
    content = content.replace('from src.', 'from arshai.')
    content = content.replace('import arshai.', 'import arshai.')
    
    with open(file_path, 'w') as f:
        f.write(content)

def update_all_imports(base_path):
    """Update imports in all Python files."""
    arshai_path = base_path / "arshai"
    
    for py_file in arshai_path.glob("**/*.py"):
        try:
            update_imports_in_file(py_file)
        except Exception as e:
            print(f"Error updating {py_file}: {e}")
    
    print("✓ Updated imports")

def update_pyproject_toml(base_path):
    """Update pyproject.toml for the new structure."""
    pyproject_path = base_path / "pyproject.toml"
    
    with open(pyproject_path, 'r') as f:
        content = f.read()
    
    # Update packages
    content = content.replace(
        'packages = [\n    {include = "src"},\n    {include = "seedwork"}\n]',
        'packages = [{include = "arshai"}]'
    )
    
    # Update version
    content = content.replace('version = "0.1.0"', 'version = "0.2.0"')
    
    with open(pyproject_path, 'w') as f:
        f.write(content)
    
    print("✓ Updated pyproject.toml")

def main():
    """Main refactoring function."""
    print("Starting package structure refactoring...")
    
    base_path = create_directory_structure()
    move_interfaces(base_path)
    move_source_files(base_path)
    update_all_imports(base_path)
    update_pyproject_toml(base_path)
    
    print("\n✅ Refactoring complete!")
    print("\nNext steps:")
    print("1. Review the changes")
    print("2. Run tests to ensure everything works")
    print("3. Update documentation")
    print("4. Commit the changes")

if __name__ == "__main__":
    main()