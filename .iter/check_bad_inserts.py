"""Check for imports accidentally placed inside docstrings."""
import os

core_dirs = ['environments', 'data', 'risk', 'training', 'agents', 'evaluation']
found = False
for cd in core_dirs:
    for root, dirs, files in os.walk(cd):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for f in files:
            if not f.endswith('.py'):
                continue
            path = os.path.join(root, f)
            with open(path) as fh:
                content = fh.read()
            lines = content.split('\n')
            in_doc = False
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    in_doc = not in_doc
                if in_doc and ('import ' in stripped):
                    print(f'{path}:{i+1}: import inside docstring: {stripped}')
                    found = True

if not found:
    print('No imports found inside docstrings - all clear!')
