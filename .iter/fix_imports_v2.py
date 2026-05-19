"""
Robust batch fixer: first strips bad docstring imports, then adds them properly.
"""
import os, ast

COMMON_IMPORTS = [
    'import numpy as np',
    'import pandas as pd',
]

core_dirs = ['environments', 'data', 'risk', 'training', 'agents', 'evaluation']
total_fixed = 0

def strip_docstring_imports(lines):
    """Remove import statements that are inside docstrings (bad insertions)."""
    result = []
    in_doc = False
    changed = False
    
    for line in lines:
        stripped = line.strip()
        
        if not in_doc and (stripped.startswith('"""') or stripped.startswith("'''")):
            q = stripped[:3]
            if q in stripped[3:]:
                # Single-line docstring
                result.append(line)
                continue
            in_doc = True
            result.append(line)
            continue
        
        if in_doc:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_doc = False
                result.append(line)
                continue
            # Check if this line has an import statement
            if any(x in stripped for x in ['import numpy', 'import pandas', 'import torch', 'import warnings']):
                changed = True
                continue  # Skip this line
            result.append(line)
            continue
        
        result.append(line)
    
    return result, changed

def find_first_code_line(lines):
    """Find first line after docstrings that's not a comment."""
    in_doc = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not in_doc and (stripped.startswith('"""') or stripped.startswith("'''")):
            if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                continue
            in_doc = True
            continue
        elif in_doc:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_doc = False
                continue
            continue
        if stripped and not stripped.startswith('#'):
            return i
    return 0

def has_proper_import(content, stmt):
    """Check if import statement exists outside docstrings."""
    lines = content.split('\n')
    in_doc = False
    for line in lines:
        stripped = line.strip()
        if not in_doc and (stripped.startswith('"""') or stripped.startswith("'''")):
            if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                continue
            in_doc = True
            continue
        elif in_doc:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_doc = False
                continue
            continue
        if stripped == stmt:
            return True
    return False

for cd in core_dirs:
    for root, dirs, files in os.walk(cd):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for f in sorted(files):
            if not f.endswith('.py'):
                continue
            path = os.path.join(root, f)
            with open(path) as fh:
                content = fh.read()
            
            lines = content.split('\n')
            
            # Step 1: Strip bad docstring imports
            lines, changed = strip_docstring_imports(lines)
            
            # Step 2: Check what's still missing
            content2 = '\n'.join(lines)
            need_np = False
            need_pd = False
            
            try:
                tree = ast.parse(content2)
                imports = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.asname or alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])
                
                used = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                        used.add(node.value.id)
                
                need_np = 'np' in used and 'numpy' not in imports and not has_proper_import(content2, 'import numpy as np')
                need_pd = 'pd' in used and 'pandas' not in imports and not has_proper_import(content2, 'import pandas as pd')
            except SyntaxError:
                continue
            
            if not need_np and not need_pd and not changed:
                continue
            
            # Step 3: Add missing imports at correct position
            if need_np or need_pd:
                insert_pos = find_first_code_line(lines)
                additions = []
                if need_np:
                    additions.append('import numpy as np')
                if need_pd:
                    additions.append('import pandas as pd')
                for j, stmt in enumerate(additions):
                    lines.insert(insert_pos + j, stmt)
            
            new_content = '\n'.join(lines)
            if new_content != content:
                with open(path, 'w') as fh:
                    fh.write(new_content)
                total_fixed += 1
                actions = []
                if changed: actions.append('stripped docstring imports')
                if need_np: actions.append('added numpy')
                if need_pd: actions.append('added pandas')
                print(f'FIXED: {path} ({", ".join(actions)})')

print(f'\nTotal files fixed: {total_fixed}')
