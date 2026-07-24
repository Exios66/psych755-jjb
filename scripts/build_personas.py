#!/usr/bin/env python3
"""Build foolproof persona prompts from Prolific + Qualtrics characteristics."""

import sys

from ca_personas.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["build-personas", *sys.argv[1:]]))
