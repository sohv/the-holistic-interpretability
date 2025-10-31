import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from cnn_model import SmallCNN
import csv


def load_data(batch_size=256):
    """Return a DataLoader for the Fashion-MNIST test set."""
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return test_loader
def evaluate(model, test_loader, device='cpu'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            _, pred = torch.max(out, 1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


def compute_neuron_importance(model, test_loader, target_class=7, device='cpu', max_samples=1000):
    model.to(device)
    model.eval()
    activations = []
    labels = []
    with torch.no_grad():
        count = 0
        for data, target in test_loader:
            data = data.to(device)
            acts = model.get_fc1_activations(data)  # shape (batch, neurons)
            activations.append(acts.cpu())
            labels.append(target)
            count += data.size(0)
            if count >= max_samples:
                break
    activations = torch.cat(activations, dim=0)
    labels = torch.cat(labels, dim=0)

    # Average activation per neuron for the target class
    mask = labels == target_class
    if mask.sum() == 0:
        raise ValueError('No samples for target class in dataset subset')
    target_acts = activations[mask]
    importance = target_acts.mean(dim=0).numpy()  # shape (neurons,)
    return importance


def compute_weight_magnitude_importance(model):
    """Importance by input-column norm of fc2 weights."""
    # fc2 weight shape: (out, in)
    w = model.fc2.weight.data.cpu().numpy()
    # magnitude per input neuron: L2 norm across output neurons
    mags = np.linalg.norm(w, axis=0)
    return mags


def compute_gradient_importance(model, data_loader, target_class=7, device='cpu', max_samples=500):
    """Compute importance by average gradient magnitude of class logit wrt fc1 activations."""
    model.to(device)
    model.eval()
    grads = []
    count = 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        acts = model.get_fc1_activations(data)
        acts = acts.detach()
        acts.requires_grad_(True)
        out = model.fc2(acts)
        logits = out[:, target_class]
        # Sum logits to get scalar for backward
        logits.sum().backward(retain_graph=False)
        g = acts.grad.cpu().abs()
        grads.append(g)
        count += data.size(0)
        if count >= max_samples:
            break
    grads = torch.cat(grads, dim=0)
    importance = grads.mean(dim=0).numpy()
    return importance


def ablate_and_evaluate(model, test_loader, neuron_indices, device='cpu'):
    model.to(device)
    # Create a copy of model weights to restore later
    orig_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Zero out selected neurons by zeroing corresponding weights in fc2
    # fc2 weight shape: (out, in)
    with torch.no_grad():
        fc2_w = model.fc2.weight  # shape (10, 128)
        # Zero columns corresponding to neurons (input dimension)
        for idx in neuron_indices:
            fc2_w[:, idx] = 0.0

    acc = evaluate(model, test_loader, device=device)

    # Restore original weights
    model.load_state_dict(orig_state)
    return acc


def random_ablation_test(model, test_loader, k=10, trials=10, device='cpu'):
    import random
    neurons = model.fc2.weight.size(1)
    results = []
    for t in range(trials):
        idxs = random.sample(range(neurons), k)
        acc = ablate_and_evaluate(model, test_loader, idxs, device=device)
        results.append(acc)
    return np.mean(results), np.std(results)


def bootstrap_ci(values, n_boot=1000, alpha=0.05):
    vals = np.array(values)
    n = len(vals)
    bs = []
    for i in range(n_boot):
        samp = np.random.choice(vals, size=n, replace=True)
        bs.append(np.mean(samp))
    lo = np.percentile(bs, 100 * (alpha/2))
    hi = np.percentile(bs, 100 * (1 - alpha/2))
    return np.mean(vals), lo, hi


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_loader = load_data()
    model = SmallCNN()
    model_path = 'models/smallcnn_fashionmnist.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError('Saved model not found. Run train_cnn_fashion_mnist.py first.')
    model.load_state_dict(torch.load(model_path, map_location=device))

    print('Baseline evaluation...')
    base_acc = evaluate(model, test_loader, device=device)
    print(f'Baseline Accuracy: {base_acc:.2f}%')

    print('Computing neuron importance for target class 7...')
    importance = compute_neuron_importance(model, test_loader, target_class=7, device=device)
    topk = np.argsort(importance)[-10:][::-1]
    print('Top-10 neurons by average activation for class 7:', topk)

    print('Ablating top-10 neurons and evaluating...')
    acc_ablate = ablate_and_evaluate(model, test_loader, topk, device=device)
    print(f'Accuracy after top-10 ablation: {acc_ablate:.2f}%')

    print('Random ablation baseline...')
    rand_mean, rand_std = random_ablation_test(model, test_loader, k=10, trials=20, device=device)
    print(f'Random ablation (k=10) mean acc: {rand_mean:.2f}% (+/- {rand_std:.2f}%)')

    # Extended experiments: per-digit ablation and k-sweep
    ks = [1, 5, 10, 20, 50]
    digits = list(range(10))
    results = []
    # Precompute weight-magnitude importance
    weight_importance = compute_weight_magnitude_importance(model)

    # Precompute gradient importance using a subset
    grad_importance = compute_gradient_importance(model, test_loader, device=device)

    for digit in digits:
        # activation-based importance
        act_importance = compute_neuron_importance(model, test_loader, target_class=digit, device=device)
        for k in ks:
            # activation top-k
            topk_act = np.argsort(act_importance)[-k:][::-1]
            acc_act = ablate_and_evaluate(model, test_loader, topk_act, device=device)

            # weight-magnitude top-k
            topk_w = np.argsort(weight_importance)[-k:][::-1]
            acc_w = ablate_and_evaluate(model, test_loader, topk_w, device=device)

            # gradient-saliency top-k
            topk_g = np.argsort(grad_importance)[-k:][::-1]
            acc_g = ablate_and_evaluate(model, test_loader, topk_g, device=device)

            # random baseline (multiple trials for bootstrap)
            random_trials = []
            neurons = model.fc2.weight.size(1)
            import random
            for t in range(30):
                idxs = random.sample(range(neurons), k)
                random_trials.append(ablate_and_evaluate(model, test_loader, idxs, device=device))
            rand_mean = float(np.mean(random_trials))
            rand_std = float(np.std(random_trials))
            rand_mean_bs, rand_lo, rand_hi = bootstrap_ci(random_trials, n_boot=500)

            print(f'Digit {digit}, k={k}: act={acc_act:.2f}%, weight={acc_w:.2f}%, grad={acc_g:.2f}%, rand={rand_mean:.2f}%')
            results.append((digit, k, 'activation', acc_act, rand_mean, rand_std, rand_lo, rand_hi))
            results.append((digit, k, 'weight', acc_w, rand_mean, rand_std, rand_lo, rand_hi))
            results.append((digit, k, 'gradient', acc_g, rand_mean, rand_std, rand_lo, rand_hi))

    # Save results to CSV
    os.makedirs('results', exist_ok=True)
    csv_path = os.path.join('results', 'ablation_sweep.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['digit', 'k', 'method', 'ablated_acc', 'random_mean_acc', 'random_std', 'random_ci_lo', 'random_ci_hi'])
        for row in results:
            writer.writerow(row)
    print(f'Extended ablation sweep results saved to {csv_path}')

if __name__ == '__main__':
    main()