import argparse
import sys
from bacon.tools.distill import add_distill_parser


def main():
    parser = argparse.ArgumentParser(
        prog='bacon',
        description='🥓 BACON - A Neural-Symbolic Network for Decision Making',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(
        title='available BACON tools',
        metavar=''
    )
    
    add_distill_parser(subparsers)
    
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)

if __name__ == '__main__':
    sys.exit(main())
