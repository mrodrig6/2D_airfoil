#!/usr/bin/env bash
# Week 1 Airfoil Exam Practice.  The file name intentionally matches the
# course handout command requested by the instructor.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec python3 course/build_week1_practice_exam.py
