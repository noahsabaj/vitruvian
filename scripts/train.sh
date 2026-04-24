#!/usr/bin/env bash
# Thin wrapper — the real CLI lives in `vitruvian.cli.train`.
exec uv run vit-train "$@"
