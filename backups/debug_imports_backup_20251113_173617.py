#!/usr/bin/env python3
"""
Debug version to check what's happening with Ax imports
"""

import sys
import os

# Add debugging before imports
print("=== Import Debug ===")
print(f"Python path: {sys.path[:3]}")
print(f"Current dir: {os.getcwd()}")

print("Testing Ax imports before any module loading...")
try:
    from ax.service.ax_client import AxClient
    print("✓ AxClient import works")
except Exception as e:
    print(f"✗ AxClient import failed: {e}")

print("\nNow testing bayesian_recommender import...")
try:
    import bayesian_recommender
    print(f"✓ bayesian_recommender imported, AX_AVAILABLE: {bayesian_recommender.AX_AVAILABLE}")
except Exception as e:
    print(f"✗ bayesian_recommender import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nNow testing experiment import...")
try:
    import experiment
    print("✓ experiment imported successfully")
except Exception as e:
    print(f"✗ experiment import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDebug complete")