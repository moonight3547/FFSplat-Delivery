"""Delivery module A: original pose/split policy with parallel pixel-identical I/O."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.stage1_prepare_parallel import main

if __name__ == '__main__':
    main()
