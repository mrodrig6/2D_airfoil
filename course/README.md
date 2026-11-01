# Airfoil CFD course materials

This directory contains an optional open-tool practice exercise and a 30-minute, proctored CFD interpretation practical.

- `week1_airfoil_practice.tex`: optional preparation exercise.
- `week2_airfoil_practical.tex`: proctored trimmed two-condition design practical.
- `build_week1_subcritical_figures.py`: Week 1 NACA 2412 incidence-sweep input.
- `week1_figures/`: subcritical geometry, polar, and pressure outputs.
- `build_week2_trimmed_optimization.py`: payload-trimmed, two-condition JAX-adjoint optimization study for Week 2.
- `week2_exam_*.fv`: case decks runnable through `./fv2d.sh`, without calling Python directly.
- `figures/`: committed geometry and JAX-adjoint design maps used by both handouts.

The TeX files are self-contained `article` documents and can be compiled from the repository root with:

```bash
pdflatex -output-directory=course course/week1_airfoil_practice.tex
pdflatex -output-directory=course course/week2_airfoil_practical.tex
```

To regenerate the committed study figures, install the repository's JAX extra and run:

```bash
python course/build_airfoil_geometry_figure.py
python course/build_jax_design_maps.py
python course/build_week1_subcritical_figures.py
python course/build_week2_trimmed_optimization.py
```

The three Week 2 candidate fields can instead be generated through the Bash launcher:

```bash
mkdir -p course/week2_figures
./fv2d.sh solve @course/week2_exam_a
./fv2d.sh solve @course/week2_exam_b
./fv2d.sh solve @course/week2_exam_c
./fv2d.sh sweep @course/week2_exam_incidence_sweep
```

The JAX maps are deliberately coarse design-region maps at fixed thickness and camber-crest location. They support interpretation of competing lift, drag, and shock constraints; they are not a high-fidelity optimum or a certification analysis.

Week 1 deliberately uses a separate, subcritical NACA 2412 incidence-response study. It does not reuse the transonic shock-control figures or the JAX maps assessed in Week 2.

`build_week2_trimmed_optimization.py` writes its outputs to
`course/week2_trimmed_figures/`. It trims each geometry at cruise and dash,
runs two JAX-adjoint drag optimizations, and writes the design maps, dash fields,
and `candidate_summary.csv`. Inspect the results before compiling the Week 2 PDF.
