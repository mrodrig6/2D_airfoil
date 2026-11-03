# Airfoil CFD course materials

This directory contains an open-tool Airfoil Exam Practice and a 30-minute, proctored Airfoil Exam.

- `week1_airfoil_practice.tex`: open-tool JAX lift-trim exam practice.
- `week2_airfoil_practical.tex`: proctored trimmed two-condition Airfoil Exam.
- `build_week1_practice_exam.py`: fixed-geometry NACA 2412 polar and JAX lift-trim study.
- `week1_practice_figures/`: outputs for the Airfoil Exam Practice.
- `build_week2_trimmed_optimization.py`: payload-trimmed, two-condition JAX-adjoint optimization study for Week 2.
- `airfoil_ractice_exam.sh`: one-command Week 1 figure launcher (the filename follows the handout command).
- `airfoil_exam.sh`: one-command Week 2 figure launcher.
- `week2_exam_*.fv`: case decks runnable through `./fv2d.sh`, without calling Python directly.
- `figures/`: committed geometry and JAX-adjoint design maps used by both handouts.

The TeX files are self-contained `article` documents and can be compiled from the repository root with:

```bash
pdflatex -output-directory=course course/week1_airfoil_practice.tex
pdflatex -output-directory=course course/week2_airfoil_practical.tex
```

To regenerate the two sets of assignment figures, install the repository's JAX extra and run:

```bash
./course/airfoil_ractice_exam.sh
./course/airfoil_exam.sh
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

Week 1 deliberately uses a separate, subcritical NACA 2412 lift-trim study. It uses JAX only to update incidence for a fixed shape; the Airfoil Exam uses JAX to explore shape variables after each shape is trimmed. The light grayscale figures are intended for projection and photocopying.

`build_week2_trimmed_optimization.py` writes its outputs to
`course/week2_trimmed_figures/`. It trims each geometry at cruise and dash,
runs two JAX-adjoint drag optimizations, and writes the design maps, dash fields,
and `candidate_summary.csv`. Inspect the results before compiling the Week 2 PDF.
