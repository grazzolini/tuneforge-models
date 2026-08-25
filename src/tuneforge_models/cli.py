"""Command-line interface for build, validation, and staging."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tuneforge-models")
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--wheel", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--first", type=Path, required=True)
    validate_parser.add_argument("--second", type=Path, required=True)
    validate_parser.add_argument("--report", type=Path, required=True)
    release_parser = commands.add_parser("assemble")
    release_parser.add_argument("--build", type=Path, required=True)
    release_parser.add_argument("--output", type=Path, required=True)
    release_parser.add_argument("--repository", type=Path, default=Path.cwd())
    release_parser.add_argument("--validation-report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        from .build import build

        result = build(args.wheel, args.output)
    elif args.command == "validate":
        from .validate import validate_builds

        result = validate_builds(args.first, args.second, args.report)
    else:
        from .release import assemble

        result = assemble(args.build, args.output, args.repository, args.validation_report)
    summary = {"status": "passed", "command": args.command, "result": result}
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
