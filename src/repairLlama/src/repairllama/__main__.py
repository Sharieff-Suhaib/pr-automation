"""Allow ``python -m repairllama`` to run the CLI."""

from repairllama.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
