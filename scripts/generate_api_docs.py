#!/usr/bin/env python3
"""
Script to automatically generate API documentation from docstrings.

This script scans the arshai package and generates reStructuredText files
for all modules, classes, and functions with proper docstrings.
"""

import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import List, Dict, Any


def get_package_modules(package_path: Path) -> List[str]:
    """Get all Python modules in the package."""
    modules = []
    
    for root, dirs, files in os.walk(package_path):
        # Skip __pycache__ and .pytest_cache directories
        dirs[:] = [d for d in dirs if not d.startswith('__pycache__') and not d.startswith('.')]
        
        for file in files:
            if file.endswith('.py') and not file.startswith('_'):
                # Convert file path to module path
                rel_path = Path(root).relative_to(package_path.parent)
                module_parts = list(rel_path.parts) + [file[:-3]]  # Remove .py extension
                module_name = '.'.join(module_parts)
                modules.append(module_name)
    
    return sorted(modules)


def generate_module_doc(module_name: str, output_dir: Path) -> str:
    """Generate documentation for a module."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Warning: Could not import {module_name}: {e}")
        return ""
    
    # Get module parts for file path
    parts = module_name.split('.')
    if len(parts) < 2:
        return ""
    
    # Skip top-level module
    relative_parts = parts[1:]  # Remove 'arshai' prefix
    
    if not relative_parts:
        return ""
    
    # Create directory structure
    doc_dir = output_dir
    for part in relative_parts[:-1]:
        doc_dir = doc_dir / part
        doc_dir.mkdir(exist_ok=True)
    
    # Generate documentation
    module_title = relative_parts[-1].replace('_', ' ').title()
    underline = '=' * len(module_title)
    
    content = f"{module_title}\n{underline}\n\n"
    
    # Add module docstring if available
    if module.__doc__:
        content += f"{inspect.cleandoc(module.__doc__)}\n\n"
    
    # Get classes and functions
    classes = []
    functions = []
    
    for name, obj in inspect.getmembers(module):
        if name.startswith('_'):
            continue
            
        if inspect.isclass(obj) and obj.__module__ == module_name:
            classes.append(name)
        elif inspect.isfunction(obj) and obj.__module__ == module_name:
            functions.append(name)
    
    # Add classes
    if classes:
        content += "Classes\n-------\n\n"
        for cls_name in classes:
            content += f".. autoclass:: {module_name}.{cls_name}\n"
            content += "   :members:\n"
            content += "   :undoc-members:\n"
            content += "   :show-inheritance:\n\n"
    
    # Add functions
    if functions:
        content += "Functions\n---------\n\n"
        for func_name in functions:
            content += f".. autofunction:: {module_name}.{func_name}\n\n"
    
    # Write file
    output_file = doc_dir / f"{relative_parts[-1]}.rst"
    output_file.write_text(content)
    
    return str(output_file.relative_to(output_dir))


def update_conf_py():
    """Update conf.py to ensure autodoc can find the package."""
    conf_path = Path(__file__).parent.parent / "docs_sphinx" / "conf.py"
    
    if not conf_path.exists():
        print(f"Warning: {conf_path} not found")
        return
    
    # Read current content
    content = conf_path.read_text()
    
    # Add path to package if not already present
    package_path = str(Path(__file__).parent.parent)
    insert_line = f"sys.path.insert(0, '{package_path}')"
    
    if insert_line not in content:
        # Find the imports section and add our path
        lines = content.split('\n')
        insert_index = 0
        
        for i, line in enumerate(lines):
            if line.startswith('import') or line.startswith('from'):
                insert_index = i
                break
        
        lines.insert(insert_index, f"import sys")
        lines.insert(insert_index + 1, insert_line)
        lines.insert(insert_index + 2, "")
        
        conf_path.write_text('\n'.join(lines))
        print(f"Updated {conf_path} with package path")


def main():
    """Main function."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    package_path = project_root / "arshai"
    docs_dir = project_root / "docs_sphinx"
    api_dir = docs_dir / "api"
    
    if not package_path.exists():
        print(f"Error: Package directory {package_path} not found")
        sys.exit(1)
    
    # Ensure API docs directory exists
    api_dir.mkdir(exist_ok=True)
    
    # Update conf.py
    update_conf_py()
    
    # Get all modules
    print("Scanning package for modules...")
    modules = get_package_modules(package_path)
    
    print(f"Found {len(modules)} modules")
    
    # Generate documentation for each module
    generated_files = []
    for module_name in modules:
        if module_name.startswith('arshai.'):
            file_path = generate_module_doc(module_name, api_dir)
            if file_path:
                generated_files.append(file_path)
                print(f"Generated: {file_path}")
    
    print(f"\nGenerated documentation for {len(generated_files)} modules")
    print("API documentation generation complete!")
    print(f"Files written to: {api_dir}")
    
    # Instructions for next steps
    print("\nNext steps:")
    print("1. Review generated files for accuracy")
    print("2. Add any missing module descriptions")
    print("3. Build docs with: cd docs_sphinx && make html")


if __name__ == "__main__":
    main()