Follow-up experiments to extend MNIST interpretability results.

This folder contains scripts to train a small CNN on Fashion-MNIST and run controlled ablation experiments to measure effect sizes and compare against random baselines.

Files:
- train_cnn_fashion_mnist.py: Train a small CNN and save the model.
- ablation_analysis.py: Identify top neurons for a target class and ablate them, compare to random ablations.
- requirements.txt: minimal Python dependencies for quick runs.

Usage:
1. Create a Python environment and install requirements.
2. Run `python train_cnn_fashion_mnist.py` to train and save a model.
3. Run `python ablation_analysis.py` to perform ablation experiments and print results.

Notes:
- These scripts are designed for quick iteration and educational experiments; adjust hyperparameters and dataset sizes for publication-quality results.