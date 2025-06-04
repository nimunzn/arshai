#!/usr/bin/env python3
"""
Specific migration script for petrochemical-rag project.
"""

import os
import shutil
from pathlib import Path
import re
from typing import List, Dict, Any
import json

def backup_project(project_path: Path) -> Path:
    """Create a backup of the project before migration."""
    backup_path = project_path.parent / f"{project_path.name}_backup"
    
    if backup_path.exists():
        shutil.rmtree(backup_path)
    
    shutil.copytree(project_path, backup_path)
    print(f"✓ Created backup at: {backup_path}")
    return backup_path

def remove_vendored_arshai(project_path: Path) -> None:
    """Remove the vendored arshai directory."""
    arshai_path = project_path / "arshai"
    
    if arshai_path.exists():
        print(f"Removing vendored arshai directory: {arshai_path}")
        shutil.rmtree(arshai_path)
        print("✓ Removed vendored arshai code")
    else:
        print("No vendored arshai directory found")

def update_pyproject_toml(project_path: Path) -> None:
    """Update pyproject.toml to add arshai dependency."""
    pyproject_path = project_path / "pyproject.toml"
    
    if not pyproject_path.exists():
        print("No pyproject.toml found")
        return
    
    with open(pyproject_path, 'r') as f:
        content = f.read()
    
    # Add arshai dependency
    if 'arshai' not in content:
        # Find the dependencies section
        dependencies_pattern = r'(\[tool\.poetry\.dependencies\].*?)\n([a-zA-Z])'
        
        def add_arshai_dep(match):
            deps_section = match.group(1)
            next_line = match.group(2)
            
            # Add arshai dependency
            new_deps = f'{deps_section}\narshai = "^0.2.0"\n{next_line}'
            return new_deps
        
        content = re.sub(dependencies_pattern, add_arshai_dep, content, flags=re.DOTALL)
        
        with open(pyproject_path, 'w') as f:
            f.write(content)
        
        print("✓ Added arshai dependency to pyproject.toml")

def update_imports_in_file(file_path: Path) -> bool:
    """Update imports in a single file for petrochemical-rag."""
    if not file_path.exists():
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Update arshai imports (these were using the vendored copy)
        import_mappings = [
            # Arshai imports
            (r'from arshai\.seedwork\.interfaces', r'from arshai.core.interfaces'),
            (r'from arshai\.src\.', r'from arshai.'),
            (r'import arshai\.seedwork\.interfaces', r'import arshai.core.interfaces'),
            (r'import arshai\.src\.', r'import arshai.'),
            
            # Also handle any remaining seedwork/src imports
            (r'from seedwork\.interfaces', r'from arshai.core.interfaces'),
            (r'from src\.', r'from arshai.'),
            (r'import seedwork\.interfaces', r'import arshai.core.interfaces'),
            (r'import src\.', r'import arshai.'),
        ]
        
        for old_pattern, new_pattern in import_mappings:
            content = re.sub(old_pattern, new_pattern, content)
        
        # Update any arshai.arshai patterns (double arshai)
        content = content.replace('arshai.arshai.', 'arshai.')
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def find_python_files(project_path: Path) -> List[Path]:
    """Find all Python files in the project."""
    python_files = []
    
    for root, dirs, files in os.walk(project_path):
        # Skip backup directories and common excludes
        dirs[:] = [d for d in dirs if not d.endswith('_backup') and d not in ['.git', '__pycache__', '.pytest_cache']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    return python_files

def update_all_imports(project_path: Path) -> Dict[str, int]:
    """Update all imports in the project."""
    python_files = find_python_files(project_path)
    
    stats = {
        'files_checked': len(python_files),
        'files_updated': 0,
        'errors': 0
    }
    
    for py_file in python_files:
        try:
            if update_imports_in_file(py_file):
                stats['files_updated'] += 1
                print(f"  Updated: {py_file.relative_to(project_path)}")
        except Exception as e:
            stats['errors'] += 1
            print(f"  Error in {py_file}: {e}")
    
    return stats

def create_migration_report(project_path: Path, stats: Dict[str, Any]) -> None:
    """Create a migration report."""
    report = {
        'migration_date': '2025-01-06',
        'project_path': str(project_path),
        'arshai_version': '0.2.0',
        'statistics': stats,
        'next_steps': [
            'Install the new arshai package: poetry install',
            'Test your application thoroughly',
            'Remove the backup directory when satisfied',
            'Consider using new plugin features'
        ]
    }
    
    report_path = project_path / 'migration_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Created migration report: {report_path}")

def main():
    """Main migration function for petrochemical-rag."""
    # Determine project path
    current_dir = Path.cwd()
    
    # Look for petrochemical-rag project
    if current_dir.name == 'petrochemical-rag':
        project_path = current_dir
    elif (current_dir / 'petrochemical-rag').exists():
        project_path = current_dir / 'petrochemical-rag'
    else:
        print("Error: Could not find petrochemical-rag project")
        print("Please run this script from the petrochemical-rag directory or its parent")
        return
    
    print("=" * 60)
    print("Petrochemical-RAG Migration to Arshai v0.2.0")
    print("=" * 60)
    print(f"Project path: {project_path}")
    
    # Step 1: Create backup
    print("\n1. Creating backup...")
    backup_path = backup_project(project_path)
    
    # Step 2: Update pyproject.toml
    print("\n2. Updating pyproject.toml...")
    update_pyproject_toml(project_path)
    
    # Step 3: Remove vendored arshai
    print("\n3. Removing vendored arshai...")
    remove_vendored_arshai(project_path)
    
    # Step 4: Update imports
    print("\n4. Updating imports...")
    stats = update_all_imports(project_path)
    
    # Step 5: Create report
    print("\n5. Creating migration report...")
    create_migration_report(project_path, stats)
    
    # Summary
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"Files checked: {stats['files_checked']}")
    print(f"Files updated: {stats['files_updated']}")
    print(f"Errors: {stats['errors']}")
    print(f"Backup created: {backup_path}")
    
    print("\nNEXT STEPS:")
    print("1. Install the new arshai package:")
    print("   cd", project_path)
    print("   poetry install")
    print()
    print("2. Test your application:")
    print("   poetry run python app/websocket.py")
    print()
    print("3. If everything works, remove the backup:")
    print(f"   rm -rf {backup_path}")
    print()
    print("4. Consider using new arshai features:")
    print("   - Plugin system for extensions")
    print("   - Hook system for customization")
    print()
    print("✅ Migration complete!")

if __name__ == "__main__":
    main()