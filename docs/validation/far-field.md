# Far-field sensitivity

This retained study varies the outer boundary at a fixed mesh setting. It measures domain sensitivity separately from mesh refinement.

## How far away the far field has to be

The legacy used 50 chords because it imposed the whole free-stream state at the
boundary, which reflects. This code uses a characteristic condition, so the
question is open — and it is a measurement, not a guess.

NACA 0012 at $M_\infty = 0.5$, $\alpha = 2^\circ$, 96 × 40:

| far field | $c_\ell$ | $c_d$ | iterations |
|---|---|---|---|
| 8 c | 0.24327 | 0.02308 | 6 201 |
| 12 c | 0.24565 | 0.02339 | 5 872 |
| **20 c** | **0.24711** | **0.02380** | **5 462** |
| 30 c | 0.24743 | 0.02413 | 5 189 |
| 50 c | 0.24722 | 0.02456 | 5 044 |

$c_\ell$ moves by **1.7% across the whole range**, and the default 20 chords is
within **0.05%** of 50. Even 12 chords is within 0.6%. The provisional default
is therefore confirmed: the domain can be 2.5× smaller than the legacy's, but the table does not show an iteration saving: the 8-chord run takes 6,201 iterations and the 50-chord run takes 5,044. These timings and loads apply to the reported fixed mesh settings.
