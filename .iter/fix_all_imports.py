"""
Comprehensive batch fix for missing imports in core package files.
Checks for used-but-not-imported names and adds them properly.
"""
import os
import ast
import re

# Map of short names to their full import statements
COMMON_IMPORTS = {
    'np': 'import numpy as np',
    'pd': 'import pandas as pd',
    'datetime': 'from datetime import datetime',
    'timedelta': 'from datetime import timedelta',
    'warnings': 'import warnings',
}

core_dirs = ['environments', 'data', 'risk', 'training', 'agents', 'evaluation']
total_fixes = 0

def get_insert_position(lines):
    """Find the first position after docstring and module-level comments to insert imports."""
    in_docstring = False
    docstring_delim = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Track docstring state
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if stripped.count('"""' if '"""' in stripped else "'''") >= 2 and len(stripped) > 3:
                    # Single-line docstring like """blah"""
                    continue
                in_docstring = True
                docstring_delim = '"""' if '"""' in stripped else "'''"
                continue
            if stripped and not stripped.startswith('#') and not stripped.startswith('"') and not stripped.startswith("'"):
                return i
        else:
            if docstring_delim in stripped:
                in_docstring = False
                docstring_delim = None
                continue
    
    return len(lines)  # Fallback

for cd in core_dirs:
    for root, dirs, files in os.walk(cd):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for f in files:
            if not f.endswith('.py'):
                continue
            path = os.path.join(root, f)
            with open(path) as fh:
                content = fh.read()
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                print(f'SKIP (syntax error): {path}: {e}')
                continue

            # Collect used names from function call attributes and annotations
            names_used = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    names_used.add(node.value.id)
                # Check for unqualified name usage in annotations
                if isinstance(node, ast.AnnAssign) and isinstance(node.annotation, ast.Name):
                    names_used.add(node.annotation.id)
                if isinstance(node, ast.FunctionDef):
                    for dec in node.decorator_list:
                        if isinstance(dec, ast.Name):
                            names_used.add(dec.id)
                if isinstance(node, ast.Name):
                    # Check if it's used in a non-Attribute context (simple name ref)
                    pass  # Too many false positives

            # Collect actual imports
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname or alias.name.split('.')[0]
                        imports.add(name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])

            # Find what's missing
            lines = content.split('\n')
            insert_pos = get_insert_position(lines)
            
            added = []
            for short_name, import_stmt in COMMON_IMPORTS.items():
                if short_name in names_used and short_name not in imports:
                    # Check if the import is already in the file
                    if import_stmt not in content:
                        added.append(import_stmt)

            if not added:
                continue

            new_lines = list(lines)
            for j, stmt in enumerate(added):
                new_lines.insert(insert_pos + j, stmt)

            new_content = '\n'.join(new_lines)
            if new_content != content:
                with open(path, 'w') as fh:
                    fh.write(new_content)
                total_fixes += 1
                print(f'FIXED: {path}')
                for stmt in added:
                    print(f'  -> added: {stmt}')

print(f'\nTotal files fixed: {total_fixes}')
