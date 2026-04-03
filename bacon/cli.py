import argparse
import io
import sys


def _ensure_utf8_stdout():
    """Reconfigure stdout/stderr to UTF-8 so emojis survive on Windows."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    elif sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding='utf-8', errors='replace',
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding='utf-8', errors='replace',
        )


def main():
    _ensure_utf8_stdout()

    parser = argparse.ArgumentParser(
        prog='bacon',
        description='BACON - A Neural-Symbolic Network for Decision Making',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        title='commands',
        dest='command',
        metavar='<command>',
    )

    # Register subcommands — add new ones here.
    from bacon.tools.distill import add_distill_parser
    from bacon.tools.demo import add_demo_parser

    add_distill_parser(subparsers)
    add_demo_parser(subparsers)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
