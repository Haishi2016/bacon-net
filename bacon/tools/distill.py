#!/usr/bin/env python
"""
BACON Model Distillation CLI

This tool distills trained BACON models into standalone, self-contained Python code
that can perform inference without the BACON framework.

Usage:
    python -m bacon.distill <json_file> <output_file> [options]
    
Examples:
    # Basic distillation
    python -m bacon.distill model.json inference.py
    
    # Specify aggregator type
    python -m bacon.distill model.json inference.py --aggregator lsp.half_weight
    
    # Run the generated file
    python inference.py 0.5 0.7 0.3 0.8 ...
"""

import argparse
import sys
import os
from bacon.utils import distill_bacon_to_code


def add_distill_parser(subparsers):
    """Add distill subcommand parser."""
    parser = subparsers.add_parser(
        'distill',
        help='Distill a BACON model to standalone Python code',
        description='Distill a BACON model to standalone Python code',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m bacon distill model.json inference.py
  python -m bacon distill model.json inference.py --aggregator lsp.half_weight
  python -m bacon distill heart_disease_tree_structure.json heart_disease_model.py
        """
    )
    
    parser.add_argument(
        'json_file',
        help='Path to the JSON file containing the BACON model structure'
    )
    
    parser.add_argument(
        'output_file',
        help='Path where the generated Python code will be saved'
    )
    
    parser.add_argument(
        '--aggregator',
        '-a',
        default='lsp.half_weight',
        choices=['lsp.half_weight', 'lsp.full_weight', 'math.arithmetic', 'math.geometric'],
        help='Type of aggregator used in the model (default: lsp.half_weight)'
    )
    
    parser.add_argument(
        '--mode',
        '-m',
        default='instance',
        choices=['instance', 'batch'],
        help='Generation mode: instance (zero dependencies, per-sample) or batch (NumPy, vectorized) (default: instance)'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.set_defaults(func=run_distill)
    return parser


def run_distill(args):
    """Execute distill command."""
    # Check if input file exists
    if not os.path.exists(args.json_file):
        print(f"❌ Error: Input file not found: {args.json_file}")
        return 1
    
    # Check if output file already exists
    if os.path.exists(args.output_file):
        response = input(f"⚠️  Output file '{args.output_file}' already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0
    
    print("=" * 70)
    print("BACON MODEL DISTILLATION")
    print("=" * 70)
    print(f"Input:      {args.json_file}")
    print(f"Output:     {args.output_file}")
    print(f"Aggregator: {args.aggregator}")
    print(f"Mode:       {args.mode} ({'zero dependencies' if args.mode == 'instance' else 'NumPy required'})")
    print("=" * 70)
    print()
    
    try:
        # Perform distillation
        result_file = distill_bacon_to_code(
            args.json_file,
            args.output_file,
            args.aggregator,
            mode=args.mode
        )
        
        print()
        print("=" * 70)
        print("✅ DISTILLATION COMPLETE")
        print("=" * 70)
        print(f"Generated file: {result_file}")
        print()
        print("You can now run the standalone model:")
        print(f"  python {args.output_file} <input_value_1> <input_value_2> ...")
        print()
        print("Or use it in your code:")
        print(f"  from {os.path.basename(args.output_file).replace('.py', '')} import predict")
        print(f"  result = predict([0.5, 0.7, 0.3, ...])")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ DISTILLATION FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main():
    """Standalone entry point for backward compatibility."""
    parser = argparse.ArgumentParser(
        description='Distill a BACON model to standalone Python code',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model.json inference.py
  %(prog)s model.json inference.py --aggregator lsp.half_weight
        """
    )
    
    parser.add_argument('json_file', help='Path to the JSON file containing the BACON model structure')
    parser.add_argument('output_file', help='Path where the generated Python code will be saved')
    parser.add_argument('--aggregator', '-a', default='lsp.half_weight',
                        choices=['lsp.half_weight', 'lsp.full_weight', 'math.arithmetic', 'math.geometric'],
                        help='Type of aggregator used in the model (default: lsp.half_weight)')
    parser.add_argument('--mode', '-m', default='instance', choices=['instance', 'batch'],
                        help='Generation mode: instance (zero dependencies, per-sample) or batch (NumPy, vectorized) (default: instance)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    return run_distill(args)


if __name__ == '__main__':
    sys.exit(main())
