#!/usr/bin/env bash
# Week 2 Airfoil Exam figure study.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec python3 course/build_week2_trimmed_optimization.py
