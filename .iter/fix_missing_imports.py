"""Batch fix missing numpy/pandas imports in core package files."""
import os, ast

core_dirs = ['environments', 'data', 'risk', 'training', 'agents', 'evaluation']
fixes = []

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
            except:
                continue

            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.asname or alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])

            names_used = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    names_used.add(node.value.id)

            need_np = 'np' in names_used and 'numpy' not in imports
            need_pd = 'pd' in names_used and 'pandas' not in imports
            if not need_np and not need_pd:
                continue

            lines = content.split('\n')
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                if stripped.startswith('"""') or stripped.startswith("'''") or stripped.startswith('from __future__'):
                    continue
                insert_idx = i
                break

            new_lines = list(lines)
            additions = []
            if need_np:
                additions.append('import numpy as np')
            if need_pd:
                additions.append('import pandas as pd')
            
            for j, add in enumerate(additions):
                new_lines.insert(insert_idx + j, add)

            new_content = '\n'.join(new_lines)
            if new_content != content:
                with open(path, 'w') as fh:
                    fh.write(new_content)
                fixes.append((path, need_np, need_pd))
                print(f'FIXED: {path} (np={need_np}, pd={need_pd})')

print(f'\nTotal files fixed: {len(fixes)}')
