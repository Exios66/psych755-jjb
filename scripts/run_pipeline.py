#!/usr/bin/env python3
"""Thin wrapper so the pipeline can be run without package install."""

from ca_personas.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
