import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="networkx.utils.backends")
import torch
import random
from run.run import Runner
import argparse
import os

def set_seed(seed=8420):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # For GPU determinism
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # -------------------------------
    # Parse command line arguments
    # -------------------------------
    parser = argparse.ArgumentParser(description="Run Neural Network Experiments")
    parser.add_argument("--dir_path", type=str, required=True,
                        help="Path to problem directory")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["mlp", "pinn", "dae", "all"],
                        help="Which model to run: mlp, pinn, dae, or all")
    parser.add_argument("--do_plot", type=int, default=1,
                        help="1 to plot losses after run, 0 to skip plotting")

    args = parser.parse_args()
    dir_path = args.dir_path
    mode = args.mode.lower()
    do_plot = bool(args.do_plot)

    # -------------------------------
    # Validate path
    # -------------------------------
    if not os.path.exists(dir_path):
        raise RuntimeError(f"❌ Provided path does not exist: {dir_path}")

    # -------------------------------
    # Set seed
    # -------------------------------
    set_seed(42)

    # -------------------------------
    # Initialize runner
    # -------------------------------
    runner = Runner(dir_path=dir_path)

    # -------------------------------
    # Run models based on mode
    # -------------------------------
    if mode in ["mlp", "all"]:
        print("▶ Running MLP model...")
        runner.run(mode="mlp")

    if mode in ["pinn", "all"]:
        print("▶ Running PINN model...")
        runner.run(mode="pinn")

    if mode in ["dae", "all"]:
        print("▶ Running DAE-HardNet model...")
        runner.run(mode="dae")

    # -------------------------------
    # Plot losses if requested
    # -------------------------------
    if do_plot:
        print("📊 Plotting losses...")
        runner._plot_losses()
        runner._plot_comparison()
        print("✅ Loss plotting done.")
    else:
        print("ℹ️ Skipping loss plotting.")