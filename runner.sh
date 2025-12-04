#!/bin/bash
# ==========================================================
# Run all models (MLP, PINN, KKT-HardNet) for all problems
# ==========================================================

# --- Activate environment ---
# source venv/bin/activate

# --- Problem directories ---
# PROBLEM_DIR="./Op_problems/problem_system_of_ode_loss" # Put your problem directory
# PROBLEM_DIR="Op_problems/problem_pde_greenberg"
PROBLEM_DIR="Op_problems/problem_toy_pde"
# PROBLEM_DIR="Op_problems/problem_population_lotka_volterra"



DO_PLOT=0

echo "Problem Directory: $PROBLEM_DIR"

# If you do not want to run a specific model just comment it out
python main.py --dir_path "$PROBLEM_DIR" --mode mlp  --do_plot $DO_PLOT

DO_PLOT=0

python main.py --dir_path "$PROBLEM_DIR" --mode pinn --do_plot $DO_PLOT

DO_PLOT=1

python main.py --dir_path "$PROBLEM_DIR" --mode dae  --do_plot $DO_PLOT


echo "Code have been executed for $PROBLEM_DIR"

