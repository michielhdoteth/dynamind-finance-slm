#!/usr/bin/env python3
"""
Restore essential imports that were removed by the automated fix script.
"""

import os
import re

def fix_environment_imports():
    """Fix imports in environment files."""
    env_files = [
        'environments/market_making.py',
        'environments/portfolio.py',
        'environments/regime_detection.py',
        'environments/single_asset.py'
    ]

    essential_imports = """import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

"""

    for file_path in env_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check if pandas is imported
                if 'import pandas as pd' not in content:
                    # Find the imports section and add pandas
                    lines = content.split('\n')
                    new_lines = []
                    imports_added = False

                    for line in lines:
                        new_lines.append(line)

                        # Add essential imports after the typing imports
                        if not imports_added and 'from typing import' in line and 'Union' in line:
                            new_lines.append('')
                            new_lines.append('import gymnasium as gym')
                            new_lines.append('import numpy as np')
                            new_lines.append('import pandas as pd')
                            new_lines.append('from gymnasium import spaces')
                            imports_added = True

                    fixed_content = '\n'.join(new_lines)

                    if fixed_content != content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(fixed_content)
                        print(f"Fixed imports in {file_path}")

            except Exception as e:
                print(f"Error fixing {file_path}: {e}")

def fix_data_imports():
    """Fix imports in data files."""
    data_files = [
        'data/data_manager.py',
        'data/preprocessors.py',
        'data/sources.py',
        'data/synthetic.py',
        'data/validators.py'
    ]

    for file_path in data_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check if essential imports are present
                if 'import numpy as np' not in content or 'import pandas as pd' not in content:
                    lines = content.split('\n')
                    new_lines = []
                    np_added = 'import numpy as np' in content
                    pd_added = 'import pandas as pd' in content

                    for line in lines:
                        new_lines.append(line)

                        # Add numpy if needed
                        if not np_added and 'from typing import' in line:
                            new_lines.append('import numpy as np')
                            np_added = True

                        # Add pandas if needed
                        if not pd_added and ('import numpy as np' in line or (np_added and 'from typing import' in line)):
                            new_lines.append('import pandas as pd')
                            pd_added = True

                    fixed_content = '\n'.join(new_lines)

                    if fixed_content != content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(fixed_content)
                        print(f"Fixed imports in {file_path}")

            except Exception as e:
                print(f"Error fixing {file_path}: {e}")

def main():
    """Fix all import issues."""
    print("Restoring essential imports...")

    fix_environment_imports()
    fix_data_imports()

    print("Import restoration completed.")

if __name__ == "__main__":
    main()