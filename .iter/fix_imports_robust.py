"""
Robust batch fixer for missing imports in core package files.
Properly handles docstrings and finds the correct insertion point.
"""
import os
import ast
import re

COMMON_IMPORTS = {
    'np': 'import numpy as np',
    'pd': 'import pandas as pd',
}

core_dirs = ['environments', 'data', 'risk', 'training', 'agents', 'evaluation']
total_fixes = 0

def find_insert_position(lines):
    """Find the first line after docstrings and module-level comments.
    Handles single-line docstrings and multi-line docstrings properly."""
    in_triple = False
    triple_char = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip shebang and encoding
        if stripped.startswith('#!'):
            continue
        if stripped.startswith('# -*-'):
            continue
        
        # Handle single-line comments
        if not in_triple and stripped.startswith('#'):
            continue
        
        # Handle empty lines
        if not stripped:
            continue
        
        # Track triple-quoted strings (docstrings)
        if not in_triple:
            for q in ['"""', "'''"]:
                if q in stripped:
                    count = stripped.count(q)
                    if count >= 2 and len(stripped) > 3:
                        # Single-line docstring like """blah"""
                        # Check the actual content isn't just the delimiters
                        pass
                    elif count == 1:
                        in_triple = True
                        triple_char = q
                        break
                    # More complex case - skip
                    break
            else:
                # Not in a docstring and this isn't a comment/empty line
                if not stripped.startswith('"') and not stripped.startswith("'"):
                    return i
        else:
            if triple_char in stripped:
                in_triple = False
                triple_char = None
    
    # If we never found a position, return position after any initial docstring
    # Try a simpler approach: find first non-docstring line
    return find_insert_simple(lines)

def find_insert_simple(lines):
    """Simpler approach: find the first import or first code line."""
    in_doc = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not in_doc and (stripped.startswith('"""') or stripped.startswith("'''")):
            in_doc = True
            # Check for single-line docstring
            if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                in_doc = False
            continue
        elif in_doc:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_doc = False
                continue
            continue
        
        if not stripped or stripped.startswith('#'):
            continue
        return i
    return 0

def get_real_imports(tree):
    """Get set of imported module short names."""
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split('.')[0]
                imports.add(name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return imports

def get_used_names(tree):
    """Get set of names used as attribute bases (like np.array, pd.DataFrame)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            names.add(node.value.id)
    return names

for cd in core_dirs:
    for root, dirs, files in os.walk(cd):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for f in sorted(files):
            if not f.endswith('.py'):
                continue
            path = os.path.join(root, f)
            with open(path) as fh:
                content = fh.read()
            
            try:
                tree = ast.parse(content)
            except SyntaxError:
                continue
            
            imports = get_real_imports(tree)
            used = get_used_names(tree)
            
            lines = content.split('\n')
            
            added = []
            for short_name, import_stmt in COMMON_IMPORTS.items():
                if short_name in used and short_name not in imports:
                    if import_stmt not in content:
                        added.append(import_stmt)
            
            if not added:
                continue
            
            # First remove any bad docstring insertions
            new_lines = []
            in_doc = False
            for line in lines:
                stripped = line.strip()
                
                # Track docstring state
                if not in_doc and (stripped.startswith('"""') or stripped.startswith("'''")):
                    q = '"""' if stripped.startswith('"""') else "'''"
                    # Check if single-line docstring
                    rest = stripped[3:]
                    if q in rest:
                        # Single-line, just keep it
                        new_lines.append(line)
                        continue
                    else:
                        # Start of multi-line docstring
                        in_doc = True
                        new_lines.append(line)
                        continue
                elif in_doc:
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        in_doc = False
                        new_lines.append(line)
                        continue
                    elif 'import ' in stripped:
                        # Skip import statements inside docstrings
                        continue
                    else:
                        new_lines.append(line)
                        continue
                
                new_lines.append(line)
            
            # Now find insertion point AFTER docstrings
            insert_pos = find_insert_simple(new_lines)
            
            # Insert missing imports
            for j, stmt in enumerate(added):
                new_lines.insert(insert_pos + j, stmt)
            
            new_content = '\n'.join(new_lines)
            if new_content != content:
                with open(path, 'w') as fh:
                    fh.write(new_content)
                total_fixes += 1
                print(f'FIXED: {path}')
                for stmt in added:
                    print(f'  + {stmt}')

print(f'\nTotal files fixed: {total_fixes}')
