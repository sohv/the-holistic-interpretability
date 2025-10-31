"""
plot_k_sweep.py

Small utility to plot k-sweep results from `results/ablation_sweep.csv`.
Generates two PNGs in `results/figs/`:
 - k_sweep_mean_accuracy.png (mean random baseline vs top-k ablated)
 - k_sweep_delta_accuracy.png (mean accuracy drop random - ablated)

Usage:
    python follow_up_experiments/plot_k_sweep.py --input results/ablation_sweep.csv --outdir results/figs

This script avoids heavy dependencies (uses only built-in csv + numpy + matplotlib).
"""

import os
import csv
import argparse
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_ablation_csv(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def aggregate_by_k(rows):
    by_k = defaultdict(list)
    for r in rows:
        # robust parsing of k
        k = int(float(r.get('k', r.get('K', r.get('k ', 0)))))
        # try canonical column names then fallbacks
        def fget(names, default=0.0):
            for n in names:
                if n in r and r[n] != '':
                    try:
                        return float(r[n])
                    except Exception:
                        pass
            return float(default)
        ablated = fget(['ablated_acc', 'ablated', 'ablated_acc '], 0.0)
        rand_mean = fget(['random_mean_acc', 'random_mean', 'random_mean_acc '], 0.0)
        rand_std = fget(['random_std', 'random_std '], 0.0)
        by_k[k].append((ablated, rand_mean, rand_std))
    return by_k


def plot_k_sweep(by_k, outdir):
    ks = sorted(by_k.keys())
    mean_abl = []
    mean_rand = []
    std_rand = []
    for k in ks:
        arr = np.array(by_k[k])
        mean_abl.append(arr[:,0].mean())
        mean_rand.append(arr[:,1].mean())
        std_rand.append(arr[:,2].mean())

    os.makedirs(outdir, exist_ok=True)

    # Plot 1: mean accuracy vs k
    plt.figure(figsize=(6,4))
    plt.plot(ks, mean_rand, marker='o', label='random baseline mean')
    plt.plot(ks, mean_abl, marker='o', label='top-k ablated')
    mean_rand_arr = np.array(mean_rand)
    std_rand_arr = np.array(std_rand)
    plt.fill_between(ks, mean_rand_arr-std_rand_arr, mean_rand_arr+std_rand_arr, color='gray', alpha=0.2)
    plt.xlabel('k (number of neurons)')
    plt.xticks(ks)
    plt.ylabel('Accuracy (%)')
    plt.title('Mean accuracy vs k (ablated top-k vs random baseline)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fn1 = os.path.join(outdir, 'k_sweep_mean_accuracy.png')
    plt.savefig(fn1)
    plt.close()

    # Plot 2: mean delta = random - ablated
    deltas = np.array(mean_rand) - np.array(mean_abl)
    plt.figure(figsize=(6,4))
    plt.plot(ks, deltas, marker='o', color='C2')
    plt.xlabel('k (number of neurons)')
    plt.xticks(ks)
    plt.ylabel('Accuracy drop (random - ablated) (%)')
    plt.title('Mean accuracy drop vs k')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fn2 = os.path.join(outdir, 'k_sweep_delta_accuracy.png')
    plt.savefig(fn2)
    plt.close()

    return fn1, fn2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='results/ablation_sweep.csv', help='Input CSV path')
    parser.add_argument('--outdir', '-o', default='results/figs', help='Output directory for figures')
    args = parser.parse_args()

    rows = load_ablation_csv(args.input)
    if not rows:
        raise SystemExit('Input CSV is empty or not found: %s' % args.input)
    by_k = aggregate_by_k(rows)
    fn1, fn2 = plot_k_sweep(by_k, args.outdir)
    print('Saved figures:', fn1, fn2)


if __name__ == '__main__':
    main()
