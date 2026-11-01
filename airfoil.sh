#!/usr/bin/env bash
#
# airfoil.sh -- one entry point for the 2D finite-volume airfoil code.
#
# Nothing has to be set up first.  This finds a Python, puts the repository on
# the import path if the package is not installed, and runs what you asked for
# from wherever you happen to be:
#
#     ./airfoil.sh mesh airfoil=0012 n_surface=192 figure=mesh.png
#     ./airfoil.sh run cases/transonic.json
#
# `./airfoil.sh help` lists everything.  `./airfoil.sh install` is optional.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SELF="$(basename -- "${BASH_SOURCE[0]}")"

die() { printf '%s: %s\n' "$SELF" "$*" >&2; exit 2; }

# --------------------------------------------------------------------------
# Python
# --------------------------------------------------------------------------
# Whatever environment you have loaded is the one used.  Set AIRFOIL_PYTHON to
# override, e.g. AIRFOIL_PYTHON=python3.12 ./airfoil.sh mesh
find_python() {
    if [[ -n ${AIRFOIL_PYTHON:-} ]]; then
        command -v "$AIRFOIL_PYTHON" >/dev/null 2>&1 || die \
            "AIRFOIL_PYTHON=$AIRFOIL_PYTHON is not an executable on PATH"
        printf '%s\n' "$AIRFOIL_PYTHON"
        return
    fi
    local candidate
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    die "no python3 on PATH.  Load your environment first, or set AIRFOIL_PYTHON."
}

# An installed package wins; otherwise the repository root goes on the path, so
# the code runs straight out of a clone with no install step and from any
# working directory.  `install` is a convenience, not a prerequisite.
setup_import_path() {
    if ! "$PY" -c 'import src' >/dev/null 2>&1; then
        export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    fi
    if ! "$PY" -c 'import src' >/dev/null 2>&1; then
        printf '%s: cannot import the solver package.  ' "$SELF" >&2
        "$PY" -c 'import src' 2>&1 | tail -3 >&2
        printf '\nTry:  ./%s install\n' "$SELF" >&2
        exit 2
    fi
}

# --------------------------------------------------------------------------
# Case files
# --------------------------------------------------------------------------
# A case is a JSON object whose keys are the names the Python API uses.  Run it
# with `run`; any `name=value` after the file overrides that key:
#
#     ./airfoil.sh run cases/transonic.json alpha=2.0 cp=cp.png
#
# The key "description" is for people and is not passed on.
CASE_DIR="$ROOT/cases"

list_cases() {
    local path found=0
    for path in "$CASE_DIR"/*.json; do
        [[ -e $path ]] || continue
        found=1
        printf '  %-28s %s\n' "${path#"$ROOT"/}" "$(case_description "$path")"
    done
    (( found )) || printf '  (none in %s)\n' "${CASE_DIR#"$ROOT"/}"
}

case_description() {
    local py=${PY:-$(find_python)}
    "$py" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("description", ""))' "$1"
}

# Expand every case file into `name=value` tokens, leaving other arguments as
# they are.  EXPANDED is set as a side effect; bash functions cannot return arrays.
expand_cases() {
    EXPANDED=()
    local token text line
    for token in ${@+"$@"}; do
        if [[ $token != *.json ]]; then
            EXPANDED+=("$token")
            continue
        fi
        [[ -f $token ]] || die "no case file $token.  Available cases:
$(list_cases)"
        text=$("$PY" -c '
import json, sys
case = json.load(open(sys.argv[1]))
if not isinstance(case, dict):
    sys.exit("a case file holds one JSON object of name: value pairs")
for key, value in case.items():
    if key == "description":
        continue
    if isinstance(value, bool):
        value = "yes" if value else "no"
    elif isinstance(value, list):
        value = ",".join(str(item) for item in value)
    print(f"{key}={value}")
' "$token") || die "$token is not a readable case file"
        while IFS= read -r line; do EXPANDED+=("$line"); done <<< "$text"
    done
}

# --------------------------------------------------------------------------
# Argument translation
# --------------------------------------------------------------------------
# `key=value` becomes `--key-with-dashes value`, so the names you type are the
# names the Python API uses -- one vocabulary instead of two.  Real flags and
# bare positionals pass through untouched.
#
# TRANSLATED is set as a side effect; bash functions cannot return arrays.
translate() {
    expand_cases ${@+"$@"}
    set -- ${EXPANDED[@]+"${EXPANDED[@]}"}
    local out=() token key value
    for token in ${@+"$@"}; do
        if [[ $token == -* || $token != *=* ]]; then
            out+=("$token")
            continue
        fi
        key=${token%%=*}
        value=${token#*=}
        case $key in
            naca|section|airfoil)              key=airfoil ;;
            ns|n_surface)                      key=n-surface ;;
            nr|n_radial)                       key=n-radial ;;
            R|radius|far_field_radius)         key=far-field-radius ;;
            h1|first_cell_height)              key=first-cell-height ;;
            # NOT `M`: in a compressible-flow code M is the Mach number, and
            # nothing else here may claim that letter
            mach)                              key=mach ;;
            a|aoa|alpha)                       key=alpha ;;
            *)                                 key=${key//_/-} ;;
        esac
        case $key in
            normals|check)
                case $value in
                    yes|true|on|1)  out+=("--$key"); continue ;;
                    no|false|off|0) continue ;;
                esac ;;
        esac
        # `--key=value`, not `--key value`: a value that starts with a minus --
        # a negative coordinate in `window=-0.6,2.6,-0.9,0.9`, say -- looks like
        # another flag when it is a separate argument, and argparse rejects it.
        out+=("--$key=$value")
    done
    TRANSLATED=(${out[@]+"${out[@]}"})
}

python_command() {
    local subcommand=$1; shift
    translate "$@"
    exec "$PY" -m src "$subcommand" ${TRANSLATED[@]+"${TRANSLATED[@]}"}
}

# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
cmd_help() {
    cat <<EOF
$SELF -- 2D finite-volume airfoil code

Usage: ./$SELF <command> [arguments]

Getting started
  install            install the package and its optional extras with pip
  check              report the Python and which extras are installed
  list               list the case files in cases/

Running
  run CASE.json      solve the operating point in a case file and report the loads
  sweep [...]        solve a range of incidences on one mesh
  sensitivity [...]  exact design gradients by the adjoint (needs JAX)
  adapt [...]        solve, refine where the error is, and solve again
  geometry [...]     inspect a section; no mesh, no flow
  mesh [...]         generate or read a mesh and report its quality
  flow [...]         print the non-dimensional free-stream state
  bench [...]        time the backends against each other
  test [...]         run the test suite
  verify [--fast]    everything the project checks about itself: tests, lint,
                     format and spelling.  '--fast' skips the larger meshes.
  docs               print where the documentation is

Arguments are 'name=value', using the same names as the Python API:

  ./$SELF geometry airfoil=2412 figure=section.png normals=yes
  ./$SELF geometry airfoil=my_design.dat dat=resampled.dat
  ./$SELF mesh airfoil=0012 n_surface=192 n_radial=64 figure=mesh.png
  ./$SELF mesh mesh=external.su2
  ./$SELF flow mach=0.8 alpha=1.25
  ./$SELF run cases/transonic.json mach=0.8 alpha=1.25 cp=cp.png figure=mach.png
  ./$SELF run cases/transonic.json flux=hllc convergence=hist.png
  ./$SELF sweep cases/subcritical.json alphas=-4:8:2 figure=polar.png
  ./$SELF sensitivity cases/subcritical.json objective=cl
  ./$SELF sensitivity cases/subcritical.json variables=alpha,camber check=yes
  ./$SELF adapt cases/transonic.json cycles=6 figure=adapted.png history=loads.png

A case file is a JSON object of name: value pairs, and a name=value argument
after it overrides that value.  The file goes where the arguments go.  A command
reports any key it does not use, so mesh takes cases/grid.json, not
cases/transonic.json, whose operating point only run, sweep and the other
solvers read:

  ./$SELF mesh cases/grid.json
  ./$SELF mesh cases/grid.json n_radial=96

Aliases: naca=airfoil, ns=n_surface, nr=n_radial, R=far_field_radius,
h1=first_cell_height, aoa=alpha.  Ordinary --flags work too, so anything
'python -m src --help' documents is still available.  'M' is deliberately NOT
an alias: in this code M always means Mach number.

Cases (cases/*.json):
$(list_cases)

Environment
  AIRFOIL_PYTHON        the interpreter to use (default: python3, then python)

No install is required: if the package is not importable, the repository root
is put on PYTHONPATH automatically.  Documentation is in docs/ -- start with
docs/README.md for the index.
EOF
}

cmd_install() {
    local extras=${1:-all}
    printf '%s: installing with pip into %s\n' "$SELF" "$("$PY" -c 'import sys; print(sys.prefix)')"
    "$PY" -m pip install -e "$ROOT[$extras]"
    printf '\n%s: done.  Checking it worked:\n\n' "$SELF"
    setup_import_path
    "$PY" -m src check
}

cmd_test() {
    translate "$@"
    # Only default to the whole tree when no path was given, so
    # `./airfoil.sh test tests/test_mesh.py` does not quietly run everything.
    local target=("$ROOT")
    local arg
    for arg in ${TRANSLATED[@]+"${TRANSLATED[@]}"}; do
        if [[ -e $arg || $arg == *::* ]]; then
            target=()
            break
        fi
    done
    exec "$PY" -m pytest ${target[@]+"${target[@]}"} ${TRANSLATED[@]+"${TRANSLATED[@]}"}
}

# Everything the project checks about itself, in one command.
#
# There is no CI by design: this is a teaching code with one author, and a push
# waiting on a queue is a worse feedback loop than a terminal.  The checks moved
# here, where they run on the machine the code is used on and the author decides
# when.  Each step is skipped with a note rather than failing if its tool is
# missing, because a missing linter should not look like a broken solver.
cmd_verify() {
    local fast=0
    [[ ${1:-} == --fast ]] && fast=1
    local failed=() skipped=()

    step() {  # step <name> <module-to-probe> <command...>
        local name=$1 probe=$2; shift 2
        if [[ -n $probe ]] && ! "$PY" -c \
            "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$probe') else 1)"
        then
            printf '\n==> %s -- SKIPPED (%s is not installed)\n' "$name" "$probe"
            skipped+=("$name")
            return 0
        fi
        printf '\n==> %s\n' "$name"
        "$@" || failed+=("$name")
    }

    step "tests (fast)" pytest "$PY" -m pytest "$ROOT" -q --fast
    if (( ! fast )); then
        step "tests (full-size meshes)" pytest \
            "$PY" -m pytest "$ROOT" -q --slow --durations=5
    fi
    step "lint" ruff "$PY" -m ruff check "$ROOT"
    step "format" ruff "$PY" -m ruff format --check "$ROOT"
    step "spelling" codespell_lib "$PY" -m codespell_lib \
        "$ROOT/src" "$ROOT/tests" "$ROOT/docs" "$ROOT/README.md"

    printf '\n%s\n' "----------------------------------------------------------"
    (( ${#skipped[@]} )) && printf 'skipped: %s\n' "${skipped[*]}"
    if (( ${#failed[@]} )); then
        printf 'FAILED:  %s\n' "${failed[*]}"
        return 1
    fi
    printf 'all checks passed\n'
}

cmd_docs() {
    cat <<EOF
docs/README.md                 the index of every documentation page
docs/theory/euler.md           the equations, the non-dimensionalisation and the free stream
docs/method/meshing.md         the O- and C-grid generator
docs/running/command-line.md   the launcher and every flag
docs/running/lab-guide.md      exercises that use the code, with the answers this code gives
docs/reference/history.md      what is built, what was removed, and why
legacy/README.md               the MATLAB this was ported from, and what the port changed
README.md                      the front door: install and quick start
EOF
}

# --------------------------------------------------------------------------
main() {
    local command=${1:-help}
    if (( $# > 0 )); then shift; fi

    case $command in
        help|-h|--help|'') cmd_help; return 0 ;;
    esac

    PY=$(find_python) || exit 2   # die() runs in a subshell; check here too

    case $command in
        install)   cmd_install ${1+"$1"} ;;
        list)      printf 'Cases (cases/*.json):\n'; list_cases ;;
        docs)      cmd_docs ;;
        test)      setup_import_path; cmd_test "$@" ;;
        verify)    setup_import_path; cmd_verify ${1+"$1"} ;;
        run)       setup_import_path; python_command solve "$@" ;;
        check|geometry|mesh|flow|sweep|sensitivity|adapt|bench)
                   setup_import_path; python_command "$command" "$@" ;;
        *)         printf '%s: unknown command %s\n\n' "$SELF" "$command" >&2
                   cmd_help >&2
                   exit 2 ;;
    esac
}

main "$@"
