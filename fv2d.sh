#!/usr/bin/env bash
#
# fv2d.sh -- one entry point for the 2D finite-volume airfoil code.
#
# Nothing has to be set up first.  This finds a Python, puts the repository on
# the import path if the package is not installed, and runs what you asked for
# from wherever you happen to be:
#
#     ./fv2d.sh mesh airfoil=0012 n_surface=192 figure=mesh.png
#     ./fv2d.sh geometry airfoil=my_design.dat
#
# `./fv2d.sh help` lists everything.  `./fv2d.sh install` is optional.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SELF="$(basename -- "${BASH_SOURCE[0]}")"

die() { printf '%s: %s\n' "$SELF" "$*" >&2; exit 2; }

# --------------------------------------------------------------------------
# Python
# --------------------------------------------------------------------------
# Whatever environment you have loaded is the one used.  Set FV2D_PYTHON to
# override, e.g. FV2D_PYTHON=python3.12 ./fv2d.sh mesh
find_python() {
    if [[ -n ${FV2D_PYTHON:-} ]]; then
        command -v "$FV2D_PYTHON" >/dev/null 2>&1 || die \
            "FV2D_PYTHON=$FV2D_PYTHON is not an executable on PATH"
        printf '%s\n' "$FV2D_PYTHON"
        return
    fi
    local candidate
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    die "no python3 on PATH.  Load your environment first, or set FV2D_PYTHON."
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
# Argument decks
# --------------------------------------------------------------------------
# `@name` is replaced by the `name=value` lines in cases/name.fv, so a case you
# run often lives in a file instead of in your shell history:
#
#     ./fv2d.sh mesh @transonic              # exactly the lines in that deck
#     ./fv2d.sh mesh @transonic n_radial=96  # the deck, with one value changed
#
# A deck holds exactly what you would have typed, so there is no second
# vocabulary to learn: `#` starts a comment, blank lines are ignored, and a deck
# may name another deck to build on it.  Later arguments win.
DECK_DIR="$ROOT/cases"
DECK_EXT=".fv"
DECK_MAX_DEPTH=8

list_decks() {
    local path stem found=0
    for path in "$DECK_DIR"/*"$DECK_EXT"; do
        [[ -e $path ]] || continue
        found=1
        stem=$(basename "$path" "$DECK_EXT")
        printf '  @%-13s %s\n' "$stem" "$(deck_summary "$path")"
    done
    (( found )) || printf '  (none in %s)\n' "${DECK_DIR#"$ROOT"/}"
}

deck_summary() {
    # the summary is the first comment line, which is why decks start with one
    sed -n '1{/^[[:space:]]*#/{s/^[[:space:]]*#[[:space:]]*//;p;};q;}' "$1"
}

resolve_deck() {
    local want=$1 candidate
    for candidate in \
        "$want" "$want$DECK_EXT" "$DECK_DIR/$want" "$DECK_DIR/$want$DECK_EXT"
    do
        if [[ -f $candidate ]]; then printf '%s\n' "$candidate"; return 0; fi
    done
    printf '%s: no deck named %s.  Available:\n' "$SELF" "$want" >&2
    list_decks >&2
    return 1
}

# Replace every `@deck` token with that deck's lines, recursively.
# EXPANDED is set as a side effect; bash functions cannot return arrays.
expand_decks() {
    local depth=$1; shift
    (( depth <= DECK_MAX_DEPTH )) || die \
        "decks nested more than $DECK_MAX_DEPTH deep -- does a deck include itself?"
    local out=() lines=() token path line
    for token in ${@+"$@"}; do
        if [[ $token != @* ]]; then
            out+=("$token")
            continue
        fi
        path=$(resolve_deck "${token#@}") || exit 2
        lines=()
        while IFS= read -r line || [[ -n $line ]]; do
            line=${line%%#*}                        # drop comments
            line=${line#"${line%%[![:space:]]*}"}   # trim left
            line=${line%"${line##*[![:space:]]}"}   # trim right
            [[ -n $line ]] && lines+=("$line")
        done < "$path"
        expand_decks $((depth + 1)) ${lines[@]+"${lines[@]}"}
        out+=(${EXPANDED[@]+"${EXPANDED[@]}"})
    done
    EXPANDED=(${out[@]+"${out[@]}"})
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
    expand_decks 1 ${@+"$@"}
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
            normals)
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

fvairfoil() {
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
  list               list the decks '@' expands

Running
  solve [...]        solve one operating point and report the loads
  geometry [...]     inspect a section; no mesh, no flow
  mesh [...]         generate or read a mesh and report its quality
  flow [...]         print the non-dimensional free-stream state
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
  ./$SELF solve @transonic mach=0.8 alpha=1.25 cp=cp.png figure=mach.png
  ./$SELF solve @transonic flux=hllc convergence=hist.png

A '@name' argument is replaced by the lines of cases/name.fv.  Later arguments
win, so a deck is a starting point you can override:

  ./$SELF mesh @transonic
  ./$SELF mesh @transonic n_radial=96

Aliases: naca=airfoil, ns=n_surface, nr=n_radial, R=far_field_radius,
h1=first_cell_height, aoa=alpha.  Ordinary --flags work too, so anything
'python -m src --help' documents is still available.  'M' is deliberately NOT
an alias: in this code M always means Mach number.

Decks ('@name'):
$(list_decks)

Environment
  FV2D_PYTHON        the interpreter to use (default: python3, then python)

No install is required: if the package is not importable, the repository root
is put on PYTHONPATH automatically.  Documentation is in docs/ -- start with
docs/theory.md for the formulation.
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
    # `./fv2d.sh test tests/test_mesh.py` does not quietly run everything.
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
docs/theory.md      the formulation, the non-dimensionalisation and the mesh
                    generator -- equations in LaTeX, renders on GitHub with
                    nothing to build
docs/usage.md       driving the code: the launcher, the Python API, every option
docs/roadmap.md     what is built, what is not, and what lands when
legacy/README.md    the MATLAB this was ported from, and what the port changed
README.md           the quick start
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
        list)      printf "Decks ('@name'):\n"; list_decks ;;
        docs)      cmd_docs ;;
        test)      setup_import_path; cmd_test "$@" ;;
        verify)    setup_import_path; cmd_verify ${1+"$1"} ;;
        check|geometry|mesh|flow|solve)
                   setup_import_path; fvairfoil "$command" "$@" ;;
        *)         printf '%s: unknown command %s\n\n' "$SELF" "$command" >&2
                   cmd_help >&2
                   exit 2 ;;
    esac
}

main "$@"
