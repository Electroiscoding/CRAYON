#!/usr/bin/env python3
"""
CRAYON Codebase Exporter

Exports all source code files (.py, .cu, .c, .cpp, .h, .hip) from the repository
into a single consolidated .txt file for documentation or analysis purposes.
"""

import os
from pathlib import Path
from datetime import datetime

# Configuration
REPO_ROOT = Path(__file__).parent
OUTPUT_FILE = REPO_ROOT / "CRAYON_Full_Codebase.txt"

# File extensions to include
EXTENSIONS = {'.py', '.cu', '.c', '.cpp', '.h', '.hip', '.hpp', '.cuh'}

# Directories to exclude
EXCLUDE_DIRS = {
    'venv', '.venv', 'env', '.env',
    '__pycache__', '.git', '.idea', '.vscode',
    'node_modules', 'build', 'dist', 'egg-info',
    '.eggs', '*.egg-info', 'site-packages'
}

# Files to exclude
EXCLUDE_FILES = {
    'export_codebase.py',  # Don't include this script itself
}


def should_exclude_dir(dir_name: str) -> bool:
    """Check if directory should be excluded."""
    return dir_name in EXCLUDE_DIRS or dir_name.startswith('.')


def should_include_file(file_path: Path) -> bool:
    """Check if file should be included based on extension and exclusions."""
    if file_path.name in EXCLUDE_FILES:
        return False
    if file_path.suffix.lower() not in EXTENSIONS:
        return False
    # Skip files in excluded directories
    for part in file_path.parts:
        if should_exclude_dir(part):
            return False
    return True


def get_file_header(file_path: Path, relative_path: Path) -> str:
    """Generate a header for each file section."""
    separator = "=" * 80
    return f"""
{separator}
FILE: {relative_path}
{separator}
"""


def collect_files(root: Path) -> list:
    """Collect all matching files from the repository."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter out excluded directories (modifies in-place to prevent descent)
        dirnames[:] = [d for d in dirnames if not should_exclude_dir(d)]
        
        for filename in filenames:
            file_path = Path(dirpath) / filename
            if should_include_file(file_path):
                files.append(file_path)
    
    # Sort files for consistent output
    return sorted(files)


def export_codebase():
    """Main export function."""
    print("=" * 60)
    print("CRAYON Codebase Exporter")
    print("=" * 60)
    print(f"\nScanning: {REPO_ROOT}")
    print(f"Extensions: {', '.join(sorted(EXTENSIONS))}")
    print()
    
    # Collect all files
    files = collect_files(REPO_ROOT)
    
    if not files:
        print("No matching files found!")
        return
    
    print(f"Found {len(files)} source files\n")
    
    # Statistics
    total_lines = 0
    total_bytes = 0
    file_stats = []
    
    # Build output content
    content_parts = []
    
    # Header
    header = f"""{'#' * 80}
#
# XERV CRAYON - Complete Codebase Export
#
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Total Files: {len(files)}
# Extensions: {', '.join(sorted(EXTENSIONS))}
#
{'#' * 80}

TABLE OF CONTENTS
{'=' * 40}
"""
    
    # Generate TOC
    toc_lines = []
    for i, file_path in enumerate(files, 1):
        relative = file_path.relative_to(REPO_ROOT)
        toc_lines.append(f"{i:4d}. {relative}")
    
    header += "\n".join(toc_lines)
    header += f"\n\n{'=' * 80}\nFILE CONTENTS\n{'=' * 80}\n"
    
    content_parts.append(header)
    
    # Process each file
    for file_path in files:
        relative = file_path.relative_to(REPO_ROOT)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                file_content = f.read()
            
            lines = file_content.count('\n') + 1
            bytes_count = len(file_content.encode('utf-8'))
            
            total_lines += lines
            total_bytes += bytes_count
            file_stats.append((relative, lines, bytes_count))
            
            # Add file section
            file_section = get_file_header(file_path, relative)
            file_section += file_content
            if not file_content.endswith('\n'):
                file_section += '\n'
            
            content_parts.append(file_section)
            print(f"  [OK] {relative} ({lines} lines)")
            
        except Exception as e:
            print(f"  [ERR] {relative} - Error: {e}")
    
    # Write output file
    print(f"\nWriting to: {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("".join(content_parts))
    
    # Summary
    output_size = OUTPUT_FILE.stat().st_size
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"  Files exported:  {len(files)}")
    print(f"  Total lines:     {total_lines:,}")
    print(f"  Output size:     {output_size / 1024:.2f} KB")
    print(f"  Output file:     {OUTPUT_FILE.name}")
    print("=" * 60)


if __name__ == "__main__":
    export_codebase()
