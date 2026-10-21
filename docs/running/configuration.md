# Configuration

Store repeatable CLI arguments in a case deck. Later arguments override earlier entries, including arguments expanded from a deck.

## Decks: a case in a file

A `@name` argument is replaced by the lines of `cases/name.fv`, so a case you
run often lives in a file instead of in your shell history. Later arguments win,
which makes a deck a starting point rather than a fixed case:

```bash
./fv2d.sh mesh @grid                 # exactly the lines in that deck
./fv2d.sh solve @transonic           # the same mesh, plus the operating point
./fv2d.sh solve @transonic flux=hllc # the deck, with one value changed
```

`@grid` carries only mesh keys, and `@transonic` is `@grid` plus `mach` and
`alpha`. That split is deliberate: `mesh` and `geometry` do not take an
operating point, so handing either of them `@transonic` reports
`unrecognized arguments: --mach --alpha` rather than ignoring them.

A deck holds exactly what you would have typed — `#` starts a comment, blank
lines are ignored, and a deck may name another deck to build on it
(`cases/transonic.fv` is `@grid` plus an operating point). There is no schema to
keep in step with the code, because it is argument expansion and nothing more.

The consequence is that a deck works with any command that accepts the keys in
it. Handing `mesh` a deck carrying keys it does not take produces
`unrecognized arguments` naming them, which is the intended behaviour — it is
what a typo needs too.
