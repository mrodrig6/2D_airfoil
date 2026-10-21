# Numerical fluxes

The interface flux supplies the upwinding between neighboring states. Three fluxes ship: Roe, HLLC and AUSM⁺-up.

## Choosing a flux

Three approximate Riemann solvers supply the upwinding. In a finite-volume
scheme the Riemann solver **is** the upwinding — there is no separate upwind
operator — so this is the single most consequential option after the mesh.

| `flux=` | scheme | on this problem |
|---|---|---|
| `roe` | Roe + Harten–Hyman entropy fix | **converged in the reported study**; the default, and what the legacy used |
| `hllc` | HLLC with Batten's wave speeds | **converged in the reported study**; agrees with Roe to 3%, and has no entropy-fix parameter |
| `ausmup` | AUSM⁺-up (Liou 2006) | **converges transonically**; ~12% more drag, and **stalls below about $M_\infty = 0.6$ on a fine mesh** |

Roe and HLLC both decompose the jump into waves. AUSM⁺-up splits the *flux*
instead, into a convective part and a pressure part, so it shares no machinery
with the other two — which is what makes it a useful second opinion when a
transonic result looks suspicious.

SLAU2 was also implemented, measured over 60 solves and **removed**: it does not
converge above a peak Mach of about 0.91.
→ [flux study](../validation/flux-study.md)

Implementation: `src.physics.roe_flux`, `hllc_flux` and `ausm_up_flux`; dispatch comes from `src.config.FLUXES`. The current AUSM⁺-up uses benchmarked coefficients without an all-speed cutoff parameter exposed to the user. Its historical cutoff-Mach experiments are not current configuration options. See [the flux study](../validation/flux-study.md).
