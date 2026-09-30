#!/usr/bin/env python3
"""Compatibility wrapper for the packaged semmap-wishart CLI."""

from semmap_haken.wishart_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
