"""Public module-B entry for the confirmed delivery recipe."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.delivery_entry import module_b_args
from research.train_confirmed import main

if __name__ == '__main__':
    main(module_b_args(sys.argv[1:]))
