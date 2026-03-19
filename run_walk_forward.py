#!/usr/bin/env python3
"""
Run Walk-Forward ELO Validation

This script runs the proper walk-forward ELO validation that eliminates look-ahead bias.
Use this instead of the original engine.py for valid results.
"""

from walk_forward_engine import run_walk_forward_validation

if __name__ == "__main__":
    print("="*100)
    print("WALK-FORWARD ELO VALIDATION")
    print("No Look-Ahead Bias - Proper Train/Test Splits")
    print("="*100)
    
    results = run_walk_forward_validation()
    
    print("\n" + "="*100)
    print("VALIDATION COMPLETE")
    print("="*100)
    print("Results saved to: walk_forward_elo_results.csv")
    print("Summary saved to: WALK_FORWARD_RESULTS.md")
