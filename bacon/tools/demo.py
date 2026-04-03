"""``bacon demo`` subcommand — run built-in demos."""

import argparse
import importlib
import sys
from bacon.demos import REGISTRY


def add_demo_parser(subparsers):
    """Register the ``demo`` subcommand."""
    parser = subparsers.add_parser(
        'demo',
        help='Run a built-in demo',
        description='Run a built-in BACON demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_build_epilog(),
    )

    parser.add_argument(
        'name',
        nargs='?',
        default=None,
        metavar='DEMO',
        help='Name of the demo to run (omit to list available demos)',
    )

    # demo-specific options (shared across demos where applicable)
    parser.add_argument(
        '--variables', '-n',
        type=int,
        default=3,
        help='Number of input variables (hello-world, default: 3)',
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=7,
        help='Random seed (default: 7)',
    )

    parser.set_defaults(func=run_demo, command='demo')
    return parser


def run_demo(args):
    """Dispatch to the requested demo module."""
    if args.name is None:
        _print_list()
        return 0

    if args.name not in REGISTRY:
        print(f"Unknown demo: {args.name}")
        print()
        _print_list()
        return 1

    module_path, _ = REGISTRY[args.name]
    mod = importlib.import_module(module_path)
    return mod.run(args)


def _print_list():
    print("Available demos:\n")
    for name, (_, description) in REGISTRY.items():
        print(f"  {name:20s} {description}")
    print()
    print("Run a demo:  bacon demo <name>")
    print("Example:     bacon demo hello-world --variables 4")


def _build_epilog():
    lines = ["available demos:"]
    for name, (_, desc) in REGISTRY.items():
        lines.append(f"  {name:20s} {desc}")
    lines.append("")
    lines.append("example:  bacon demo hello-world --variables 4")
    return "\n".join(lines)
