import os
import csv
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from cnn_model import SmallCNN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import random


def load_data(batch_size=256):
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return test_loader


def compute_class_mean_activations(model, data_loader, device='cpu'):
    model.to(device)
    model.eval()
    sums = None
    counts = None
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            acts = model.get_fc1_activations(data).cpu()
            for i in range(data.size(0)):
                t = int(target[i].item())
                if sums is None:
                    sums = {c: torch.zeros(acts.size(1)) for c in range(10)}
                    counts = {c: 0 for c in range(10)}
                sums[t] += acts[i]
                counts[t] += 1
    means = {c: (sums[c] / max(1, counts[c])).numpy() for c in sums}
    return means


def get_activations_and_labels(model, data_loader, device='cpu', max_samples=None):
    model.to(device)
    model.eval()
    acts_list = []
    labels_list = []
    with torch.no_grad():
        count = 0
        for data, target in data_loader:
            data = data.to(device)
            acts = model.get_fc1_activations(data).cpu()
            acts_list.append(acts)
            labels_list.append(target)
            count += data.size(0)
            if max_samples and count >= max_samples:
                break
    acts = torch.cat(acts_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    return acts, labels


def compute_neuron_clusters(activations, n_clusters=6, random_state=42):
    # activations shape: (n_samples, n_neurons) -> we cluster neurons by their profile across samples
    neuron_profiles = activations.T  # (n_neurons, n_samples)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(neuron_profiles)
    return labels, kmeans


def intervention_replace(model, data_loader, class_mean, neuron_indices, device='cpu'):
    """Replace selected neurons in fc1 activations with class mean and measure change in class probability."""
    model.to(device)
    model.eval()
    probs_before = []
    probs_after = []
    targets = []

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            acts = model.get_fc1_activations(data)
            out_before = model.fc2(acts)
            p_before = F.softmax(out_before, dim=1).cpu().numpy()

            acts_mod = acts.clone()
            # replace selected neuron columns with class mean
            for idx in neuron_indices:
                acts_mod[:, idx] = torch.tensor(class_mean[idx], device=acts_mod.device)

            out_after = model.fc2(acts_mod)
            p_after = F.softmax(out_after, dim=1).cpu().numpy()

            probs_before.append(p_before)
            probs_after.append(p_after)
            targets.append(target.numpy())

    probs_before = np.concatenate(probs_before, axis=0)
    probs_after = np.concatenate(probs_after, axis=0)
    targets = np.concatenate(targets, axis=0)

    return probs_before, probs_after, targets


def intervention_add(model, data_loader, delta_vec, neuron_indices, device='cpu'):
    """Add delta to selected neurons (insertion) and measure change in class probability."""
    model.to(device)
    model.eval()
    probs_before = []
    probs_after = []
    targets = []

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            acts = model.get_fc1_activations(data)
            out_before = model.fc2(acts)
            p_before = F.softmax(out_before, dim=1).cpu().numpy()

            acts_mod = acts.clone()
            for idx in neuron_indices:
                acts_mod[:, idx] = acts_mod[:, idx] + float(delta_vec[idx])

            out_after = model.fc2(acts_mod)
            p_after = F.softmax(out_after, dim=1).cpu().numpy()

            probs_before.append(p_before)
            probs_after.append(p_after)
            targets.append(target.numpy())

    probs_before = np.concatenate(probs_before, axis=0)
    probs_after = np.concatenate(probs_after, axis=0)
    targets = np.concatenate(targets, axis=0)

    return probs_before, probs_after, targets


def mean_delta_and_accuracy_metrics(probs_before, probs_after, targets, target_class):
    # Δp on non-targets
    mask_non_target = targets != target_class
    delta_p = probs_after[mask_non_target, target_class] - probs_before[mask_non_target, target_class]
    mean_delta = float(np.mean(delta_p))
    # accuracy change for target class (on target inputs)
    mask_target = targets == target_class
    if mask_target.sum() > 0:
        preds_before = probs_before[mask_target].argmax(axis=1)
        preds_after = probs_after[mask_target].argmax(axis=1)
        acc_before = float((preds_before == target_class).mean())
        acc_after = float((preds_after == target_class).mean())
        delta_acc = acc_after - acc_before
    else:
        acc_before = acc_after = delta_acc = float('nan')

    # effect size (Cohen's d) for delta_p
    sd = float(np.std(delta_p, ddof=1)) if delta_p.size > 1 else float('nan')
    cohens_d = mean_delta / sd if sd and not np.isnan(sd) and sd > 0 else float('nan')
    return mean_delta, delta_p, acc_before, acc_after, delta_acc, cohens_d


def bootstrap_mean(values, n_boot=1000, alpha=0.05):
    vals = np.array(values)
    bs = []
    n = len(vals)
    for i in range(n_boot):
        samp = np.random.choice(vals, size=n, replace=True)
        bs.append(np.mean(samp))
    lo = np.percentile(bs, 100 * (alpha/2))
    hi = np.percentile(bs, 100 * (1 - alpha/2))
    return float(np.mean(vals)), float(lo), float(hi)


def random_neuron_baseline(model, data_loader, k, target_class, intervention_fn, trials=200, device='cpu'):
    import random
    neurons = model.fc2.weight.size(1)
    mean_deltas = []
    for t in range(trials):
        idxs = random.sample(range(neurons), k)
        probs_before, probs_after, targets = intervention_fn(model, data_loader, np.zeros(neurons), idxs, device=device)
        mean_delta, _, _, _, _, _ = mean_delta_and_accuracy_metrics(probs_before, probs_after, targets, target_class)
        mean_deltas.append(mean_delta)
    return mean_deltas


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_loader = load_data(batch_size=256)

    model = SmallCNN()
    model_path = 'models/smallcnn_fashionmnist.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError('Saved model not found. Run train_cnn_fashion_mnist.py first.')
    model.load_state_dict(torch.load(model_path, map_location=device))

    print('Computing class mean activations...')
    means = compute_class_mean_activations(model, test_loader, device=device)

    # Choose a target class and a source class to test insertion/replacement
    target_class = 2
    source_class = 0

    # Compute importance by activation mean for target class
    # build importance scores from class mean absolute values
    class_mean = means[target_class]
    importance = np.abs(class_mean)
    topk = np.argsort(importance)[-10:][::-1]

    print('Running replacement intervention (replace top-10 neurons with class mean)...')
    probs_before, probs_after, targets = intervention_replace(model, test_loader, class_mean, topk, device=device)

    # Compute average change in target class probability for non-target inputs
    mask_non_target = targets != target_class
    delta_p = probs_after[mask_non_target, target_class] - probs_before[mask_non_target, target_class]
    mean_delta = delta_p.mean()
    print(f'Mean Δp for inserting class-{target_class} mean into top-10 (non-target inputs): {mean_delta:.4f}')

    # Save CSV of per-sample changes (limited)
    os.makedirs('results', exist_ok=True)
    csv_path = os.path.join('results', f'intervention_replace_class{target_class}_top10.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['is_target', 'p_before', 'p_after', 'delta_p'])
        for i in range(len(delta_p)):
            writer.writerow([int(mask_non_target[i]), probs_before[i, target_class], probs_after[i, target_class], probs_after[i, target_class]-probs_before[i, target_class]])

    # Plot distribution of delta_p for non-targets
    plt.figure(figsize=(6,4))
    plt.hist(delta_p, bins=50)
    plt.title(f'Distribution of Δp for replacing top-10 with class {target_class} mean')
    plt.xlabel('Δp')
    plt.ylabel('Count')
    plt.tight_layout()
    fig_path = os.path.join('results', f'intervention_replace_class{target_class}_top10_hist.png')
    plt.savefig(fig_path)
    print(f'Intervention results saved to {csv_path} and {fig_path}')

if __name__ == '__main__':
    main()
