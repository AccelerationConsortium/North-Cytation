#!/usr/bin/env python3
"""
Test regex patterns for reservoir dispensing
"""

import re

# Current patterns
patterns = [
    # Reservoir pattern (should match first)
    r'Dispensing\s+into\s+vial\s+([^\s]+)\s+from\s+reservoir\s+([0-9]+):\s+([0-9.]+)\s+mL',
    # General dispensing pattern
    r'Dispensing\s+([0-9.]+)\s+mL\s+from\s+([^\s]+)\s+to\s+([^\s]+)',
]

test_line = "2026-01-27 14:39:05,502 - INFO - Dispensing into vial 13 from reservoir 1: 5.600 mL"

print("Testing line:", test_line)
print()

for i, pattern in enumerate(patterns):
    print(f"Pattern {i+1}: {pattern}")
    match = re.search(pattern, test_line)
    if match:
        print(f"  MATCH! Groups: {match.groups()}")
    else:
        print("  No match")
    print()