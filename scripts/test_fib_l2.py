#!/usr/bin/env python
"""L2 Fib 레벨 테스트"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.context.cycle_anchor import get_fractal_fib_levels

# $60,000 근처 L0/L1/L2 레벨 확인
price = 60000
levels = get_fractal_fib_levels((59000, 61000), max_depth=2)

print(f"Price range: $59,000 ~ $61,000")
print(f"Total levels: {len(levels)}")
print()

print("L0 레벨:")
for lvl in [l for l in levels if l.depth == 0]:
    print(f"  Fib {lvl.fib_ratio:.4f} | ${lvl.price:,.0f}")

print("\nL1 레벨:")
for lvl in [l for l in levels if l.depth == 1]:
    print(f"  Fib {lvl.fib_ratio:.4f} | ${lvl.price:,.0f}")

print("\nL2 레벨:")
for lvl in [l for l in levels if l.depth == 2][:15]:
    print(f"  Fib {lvl.fib_ratio:.4f} | ${lvl.price:,.0f}")

l2_count = len([l for l in levels if l.depth == 2])
if l2_count > 15:
    print(f"  ... and {l2_count - 15} more L2 levels")
