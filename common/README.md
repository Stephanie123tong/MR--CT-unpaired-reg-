# Common Utilities for Ablation Study

This directory contains shared utilities for all ablation study versions.

## Structure

- `config.py`: Configuration system for different ablation versions
- `registration_utils.py`: MultiGradICON registration wrapper
- `README.md`: This file

## Usage

Each version folder should import from `common`:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

from config import get_config
from registration_utils import RegistrationWrapper
```

## Version Configuration

Each version enables different modules:

- **baseline**: No registration modules
- **version1**: Module 1 (Intra-domain self-supervised registration)
- **version2**: Module 2 (Cycle-registration consistency)
- **version3**: Module 3 (Pseudo-pair registration for discriminator)
- **version4**: Module 1 + Module 2
- **version5**: Module 1 + Module 3
- **version6**: Module 2 + Module 3
- **version7**: All modules

