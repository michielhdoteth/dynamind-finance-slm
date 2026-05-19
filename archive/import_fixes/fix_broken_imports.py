#!/usr/bin/env python3
"""
Fix broken imports caused by the automated fix script.
"""

import os
import re

def fix_file(file_path: str):
    """Fix broken imports in a specific file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Fix broken import patterns
        fixes = [
            # Fix "class risk.PositionLimits:" -> "class PositionLimits:"
            (r'class risk\.PositionLimits:', 'class PositionLimits:'),

            # Fix "from risk import ..., risk.PositionLimits, ..." -> "from risk import ..., PositionLimits, ..."
            (r'from risk import ([^,]+),\s*risk\.PositionLimits,', r'from risk import \1, PositionLimits,'),

            # Fix "risk.PositionLimits" -> "PositionLimits" in imports
            (r'from risk import.*risk\.PositionLimits', lambda m: m.group(0).replace('risk.PositionLimits', 'PositionLimits')),

            # Fix "risk.risk.PositionLimits" -> "risk.PositionLimits"
            (r'risk\.risk\.PositionLimits', 'risk.PositionLimits'),
        ]

        for pattern, replacement in fixes:
            if callable(replacement):
                content = re.sub(pattern, replacement, content)
            else:
                content = re.sub(pattern, replacement, content)

        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed broken imports in {file_path}")
            return True

        return False

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Fix broken imports in specific files."""
    files_to_fix = [
        'risk/risk_manager.py',
        'tests/test_qwen_advanced.py',
        'tests/test_risk_management.py',
        'tests/test_risk_simple.py',
        'training/offline_trainer.py'
    ]

    fixed_files = 0
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if fix_file(file_path):
                fixed_files += 1

    print(f"\nFixed broken imports in {fixed_files} files")

if __name__ == "__main__":
    main()