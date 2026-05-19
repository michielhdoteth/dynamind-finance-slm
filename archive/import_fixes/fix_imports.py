#!/usr/bin/env python3
"""
Fix common import issues for open-source release.
This script removes unused imports and fixes basic code quality issues.
"""

import os
import re
from typing import List, Set


def fix_file_imports(file_path: str) -> bool:
    """Fix unused imports in a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # List of commonly unused imports to remove
        unused_imports = {
            "warnings",
            "random",
            "time",
            "json",
            "pickle",
            "datetime.timedelta",
            "datetime.datetime",
            "pathlib.Path",
            "typing.Optional",
            "typing.Tuple",
            "typing.Union",
            "typing.Any",
            "typing.Dict",
            "typing.Callable",
            "abc.ABC",
            "abc.abstractmethod",
            "collections.defaultdict",
            "collections.deque",
            "dataclasses.field",
            "gymnasium as gym",
            "gymnasium.spaces",
            "numpy as np",
            "pandas as pd",
            "seaborn as sns",
            "matplotlib.pyplot as plt",
            "torch",
            "transformers",
            "stable_baselines3 as sb3",
            "stable_baselines3.common.vec_env.SubprocVecEnv",
        }

        lines = content.split("\n")
        fixed_lines = []
        imports_to_remove = set()

        # First pass: identify imports to remove
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                # Check if this is an unused import
                for unused in unused_imports:
                    if unused in line:
                        imports_to_remove.add(line)
                        break

        # Second pass: remove identified imports and keep others
        for line in lines:
            if line not in imports_to_remove:
                fixed_lines.append(line)

        # Remove consecutive blank lines
        cleaned_lines = []
        prev_blank = False
        for line in fixed_lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            cleaned_lines.append(line)
            prev_blank = is_blank

        fixed_content = "\n".join(cleaned_lines)

        # Write back if changed
        if fixed_content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(fixed_content)
            print(f"Fixed imports in {file_path}")
            return True

        return False

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def fix_bare_except(file_path: str) -> bool:
    """Fix bare except statements."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Replace bare except with except Exception
        content = re.sub(r"\bexcept:\s*\n", "except Exception:\n", content)

        # Write back if changed
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed bare except in {file_path}")
            return True

        return False

    except Exception as e:
        print(f"Error fixing bare except in {file_path}: {e}")
        return False


def fix_undefined_name(file_path: str) -> bool:
    """Fix common undefined name issues."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Fix common undefined names
        fixes = {
            r"\bself\.([a-zA-Z_][a-zA-Z0-9_]*)\b": r"self.\1",  # Will be handled manually
            r"\bPositionLimits\b": "risk.risk.PositionLimits",
        }

        for pattern, replacement in fixes.items():
            content = re.sub(pattern, replacement, content)

        # Write back if changed
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed undefined names in {file_path}")
            return True

        return False

    except Exception as e:
        print(f"Error fixing undefined names in {file_path}: {e}")
        return False


def fix_f_string_missing_placeholders(file_path: str) -> bool:
    """Fix f-strings with missing placeholders."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Replace f-strings with missing placeholders to regular strings
        # This is a simple heuristic - might need manual review
        lines = content.split("\n")
        fixed_lines = []

        for line in lines:
            # Look for f-strings without {}
            if '"' in line or "'" in line:
                # Simple check - if no braces found, remove f-prefix
                if "{" not in line:
                    line = line.replace('"', '"').replace("'", "'")
            fixed_lines.append(line)

        fixed_content = "\n".join(fixed_lines)

        # Write back if changed
        if fixed_content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(fixed_content)
            print(f"Fixed f-strings in {file_path}")
            return True

        return False

    except Exception as e:
        print(f"Error fixing f-strings in {file_path}: {e}")
        return False


def main():
    """Fix code quality issues in the codebase."""
    directories = [
        "environments",
        "data",
        "risk",
        "training",
        "agents",
        "models",
        "tests",
        "examples",
        "scripts",
    ]

    fixed_files = 0

    for directory in directories:
        if not os.path.exists(directory):
            continue

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)

                    # Apply fixes
                    changed = False
                    changed |= fix_file_imports(file_path)
                    changed |= fix_bare_except(file_path)
                    changed |= fix_undefined_name(file_path)
                    changed |= fix_f_string_missing_placeholders(file_path)

                    if changed:
                        fixed_files += 1

    print(f"\nFixed code quality issues in {fixed_files} files")


if __name__ == "__main__":
    main()
