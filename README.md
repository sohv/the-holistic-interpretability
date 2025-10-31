# The Holistic Interpretability

> **FYI**: For a detailed explanation of these experiments, check out the [blog post](https://your-blog-link.com/holistic-interpretability). The comprehensive documentation is currently **work in progress**.

## Methodology

- **Neuron Ablation**: Identifies neurons highly activated for specific digits and systematically removes them to measure their contribution to model performance
- **Activation Analysis**: Visualizes and analyzes neuron activation patterns across different layers to understand what features different neurons learn to recognize
- **Gradient Attribution**: Uses gradient-based techniques to identify which input pixels are most important for model predictions and visualizes attention maps
- **Mechanistic Interpretability**: Performs comprehensive analysis of neuron interactions, clustering, and information flow between layers to understand internal computational structure
- **Causal Tracing**: Implements advanced techniques to trace causal information flow and analyze geometric representations of concepts in activation space

## Results

### Performance Impact

| Experiment | Baseline Accuracy | Post-Intervention | Impact |
|------------|------------------|-------------------|---------|
| Neuron Ablation (Digit 7) | 97%+ | ~85% | 12% accuracy drop when removing top 10 neurons |

### Fashion-MNIST Ablation (k = 10)

The table below summarizes the ablation results on Fashion-MNIST where the top-10 neurons (by average activation for each class) were ablated and compared to a random ablation baseline (mean over multiple trials).

| Digit | Ablated Accuracy (top-10) | Random Ablation Mean | Drop |
|-------:|:-------------------------:|:--------------------:|:----:|
| 0 | 84.54% | 88.56% | 4.02% |
| 1 | 84.96% | 88.78% | 3.82% |
| 2 | 86.54% | 88.39% | 1.85% |
| 3 | 84.10% | 88.53% | 4.43% |
| 4 | 86.76% | 88.34% | 1.58% |
| 5 | 87.69% | 88.75% | 1.06% |
| 6 | 86.35% | 88.59% | 2.24% |
| 7 | 86.31% | 88.46% | 2.15% |
| 8 | 88.34% | 88.55% | 0.21% |
| 9 | 85.26% | 88.35% | 3.09% |
| **Mean** | **86.09%** | **88.53%** | **2.44%** |


### Feature Discovery

| Experiment | Key Findings |
|------------|--------------|
| **Activation Analysis** | Early layers detect basic features (edges, curves); deeper layers show digit-specific specialization |
| **Gradient Attribution** | Model focuses on distinctive digit parts (loop in '6', horizontal line in '7') |

### Network Structure Analysis

| Experiment | Discovery | Quantitative Results |
|------------|-----------|---------------------|
| **Mechanistic Interpretability** | Neuron Clustering | 6 distinct clusters identified |
| | Correlation Patterns | 607 highly correlated neuron pairs (>0.5 threshold) |
| | Cluster Specialization | Cluster 0→Digit 2, Cluster 3→Digit 0, Cluster 4→Digit 6 |
| **Causal Tracing** | Information Flow | Progressive feature abstraction across layers |
| | Feature Directions | Linear directions in activation space correspond to digit features |