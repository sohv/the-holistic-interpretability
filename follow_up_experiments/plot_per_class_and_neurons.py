"""
plot_per_class_and_neurons.py

Produces:
 - Per-class k-sweep curves from `results/ablation_sweep.csv` (one line per digit)
 - Visualizations of top-k neurons per class by reshaping fc1 weights into (channels,7,7), collapsing to 7x7 maps and upsampling to 28x28.

Saves outputs under `results/figs/`:
 - per_class_k_sweep.png
 - top_neurons_class{d}.png for each digit (if model checkpoint exists)

Usage:
    python follow_up_experiments/plot_per_class_and_neurons.py --input results/ablation_sweep.csv --model models/smallcnn_fashionmnist.pth --topk 10

If no model path is provided or the file is missing the script will still create per-class curves.
"""

import os
import csv
import argparse
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Optional torch parts
try:
    import torch
    from follow_up_experiments.cnn_model import SmallCNN
    has_torch = True
except Exception:
    has_torch = False


def plot_per_class_k_sweep(csv_path, outdir):
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    by_digit = defaultdict(lambda: defaultdict(list))
    ks_set = set()
    for r in rows:
        digit = int(float(r.get('digit', r.get('label', r.get('class', 0)))))
        k = int(float(r.get('k', r.get('K', 0))))
        ks_set.add(k)
        ablated = float(r.get('ablated_acc', r.get('ablated', np.nan)))
        rand_mean = float(r.get('random_mean_acc', r.get('random_mean', np.nan)))
        by_digit[digit]['k'].append(k)
        by_digit[digit]['ablated'].append(ablated)
        by_digit[digit]['random_mean'].append(rand_mean)
    ks = sorted(list(ks_set))
    plt.figure(figsize=(8,6))
    for digit in sorted(by_digit.keys()):
        # sort by k
        items = sorted(zip(by_digit[digit]['k'], by_digit[digit]['ablated'], by_digit[digit]['random_mean']), key=lambda x: x[0])
        k_vals = [i[0] for i in items]
        ablated = [i[1] for i in items]
        plt.plot(k_vals, ablated, marker='o', label=f'class {digit}')
    plt.xlabel('k (number of neurons)')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-class accuracy vs k (top-k ablated)')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    outfn = os.path.join(outdir, 'per_class_k_sweep.png')
    plt.savefig(outfn)
    plt.close()
    return outfn


def visualize_top_neurons(model_path, topk, outdir, device='cpu'):
    if not has_torch:
        print('torch or model module not available; skipping neuron visualizations')
        return []
    if not os.path.exists(model_path):
        print(f'model checkpoint not found at {model_path}; skipping neuron visualizations')
        return []
    # load model
    device = torch.device(device)
    model = SmallCNN()
    state = torch.load(model_path, map_location=device)
    # state may be a dict with key 'model_state' or raw state_dict
    if 'model_state_dict' in state:
        sd = state['model_state_dict']
    elif 'state_dict' in state:
        sd = state['state_dict']
    else:
        sd = state
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    # load test data to compute class mean activations
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.FashionMNIST(root='data', train=False, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    # collect fc1 activations per class
    n_neurons = model.fc1.out_features
    class_sums = {c: torch.zeros(n_neurons, device=device) for c in range(10)}
    class_counts = {c: 0 for c in range(10)}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            acts = model.get_fc1_activations(x)  # shape [B, n_neurons]
            for i, label in enumerate(y.tolist()):
                class_sums[label] += acts[i]
                class_counts[label] += 1
    class_means = {c: (class_sums[c] / class_counts[c]).cpu().numpy() for c in class_sums}

    # get fc1 weights: shape (out_features, in_features)
    fc1_w = model.fc1.weight.data.cpu().numpy()  # (n_neurons, in_features)
    # in_features = 32*7*7
    in_ch = model.conv2.out_channels
    spatial = (7,7)
    per_class_files = []
    # For each class, find top-k neurons by mean activation
    for c in range(10):
        means = class_means[c]
        top_idx = np.argsort(-means)[:topk]
        # create grid to show topk maps
        cols = min(10, topk)
        rows = int(np.ceil(topk/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*1.2, rows*1.2))
        axes = np.array(axes).reshape(-1)
        for i, idx in enumerate(top_idx):
            w = fc1_w[idx]  # length in_ch*7*7
            w3 = w.reshape(in_ch, spatial[0], spatial[1])
            # collapse channels by mean abs weight
            map7 = np.mean(w3, axis=0)
            # normalize
            map7 = map7 - map7.min()
            if map7.max() > 0:
                map7 = map7 / map7.max()
            # upsample to 28x28 via np.kron
            up = np.kron(map7, np.ones((4,4)))
            ax = axes[i]
            ax.imshow(up, cmap='viridis')
            ax.axis('off')
            ax.set_title(str(int(idx)), fontsize=8)
        # turn off any extra axes
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        plt.suptitle(f'top-{topk} neurons for class {c} (indices)')
        plt.tight_layout()
        outfn = os.path.join(outdir, f'top_neurons_class{c}.png')
        fig.savefig(outfn)
        plt.close(fig)
        per_class_files.append(outfn)
    return per_class_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='results/ablation_sweep.csv')
    parser.add_argument('--outdir', '-o', default='results/figs')
    parser.add_argument('--model', '-m', default='models/smallcnn_fashionmnist.pth')
    parser.add_argument('--topk', '-k', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    per_class_fig = plot_per_class_k_sweep(args.input, args.outdir)
    print('Saved per-class k sweep:', per_class_fig)
    files = visualize_top_neurons(args.model, args.topk, args.outdir)
    if files:
        print('Saved top-neuron visualizations:', files)
    else:
        print('No neuron visualizations generated (missing torch/model).')

if __name__ == '__main__':
    main()
