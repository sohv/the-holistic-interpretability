# References and Further Reading

## 1. Ablation Study (`ablation_mnist.ipynb`)

### Papers
- **"Visualizing and Understanding Convolutional Networks"** (Zeiler & Fergus, 2014)
  - Classic paper introducing visualization techniques and systematic occlusion experiments
  - [Paper Link](https://arxiv.org/abs/1311.2901)

- **"Understanding Neural Networks Through Deep Visualization"** (Yosinski et al., 2015)
  - Comprehensive study of feature visualization and ablation in deep networks
  - [Paper Link](https://arxiv.org/abs/1506.06579)

- **"Ablation Studies in Artificial Neural Networks"** (Meyes et al., 2019)
  - Systematic review of ablation methodologies in neural networks
  - [Paper Link](https://arxiv.org/abs/1901.08644)

### Blogs & Resources
- **Distill.pub: "Feature Visualization"** 
  - Interactive explanations of how to visualize what neural networks learn
  - [Blog Link](https://distill.pub/2017/feature-visualization/)

- **OpenAI Blog: "Microscope"**
  - Tool and techniques for neural network visualization and ablation
  - [Blog Link](https://openai.com/blog/microscope/)

## 2. Activation Analysis (`activation_analysis.ipynb`)

### Papers
- **"Do Vision Transformers See Like Convolutional Neural Networks?"** (Raghu et al., 2021)
  - Analysis of activation patterns and representation learning
  - [Paper Link](https://arxiv.org/abs/2108.08810)

- **"Network Dissection: Quantifying Interpretability of Deep Visual Representations"** (Bau et al., 2017)
  - Systematic method for understanding what individual neurons detect
  - [Paper Link](https://arxiv.org/abs/1704.05796)

- **"Understanding the Role of Individual Units in a Deep Neural Network"** (Bau et al., 2020)
  - Comprehensive framework for analyzing neuron specialization and activation patterns
  - [Paper Link](https://arxiv.org/abs/2009.05041)

### Blogs & Resources
- **Google AI Blog: "Understanding Deep Neural Networks with Activation Atlases"**
  - Techniques for visualizing and understanding activation patterns
  - [Blog Link](https://ai.googleblog.com/2019/03/understanding-deep-neural-networks.html)

- **Towards Data Science: "Neural Network Interpretability with Layer-wise Relevance Propagation"**
  - Practical guide to activation analysis techniques
  - [Blog Link](https://towardsdatascience.com/neural-network-interpretability-with-layer-wise-relevance-propagation-a2deca9772cc)

## 3. Gradient Attribution (`gradient_attribution.ipynb`)

### Papers
- **"Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps"** (Simonyan et al., 2013)
  - Foundational paper on gradient-based saliency maps
  - [Paper Link](https://arxiv.org/abs/1312.6034)

- **"Axiomatic Attribution for Deep Networks"** (Sundararajan et al., 2017)
  - Introduction of Integrated Gradients method with theoretical foundations
  - [Paper Link](https://arxiv.org/abs/1703.01365)

- **"SmoothGrad: removing noise by adding noise"** (Smilkov et al., 2017)
  - Technique for improving gradient-based attribution methods
  - [Paper Link](https://arxiv.org/abs/1706.03825)

### Blogs & Resources
- **Distill.pub: "The Building Blocks of Interpretability"**
  - Comprehensive overview of gradient-based attribution techniques
  - [Blog Link](https://distill.pub/2018/building-blocks/)

- **Google AI Blog: "SmoothGrad: Removing Noise by Adding Noise"**
  - Practical implementation of gradient smoothing techniques
  - [Blog Link](https://ai.googleblog.com/2017/06/smoothgrad-removing-noise-by-adding.html)

- **Captum Tutorial: "Introduction to Captum"**
  - PyTorch library for model interpretability including gradient methods
  - [Tutorial Link](https://captum.ai/tutorials/)

## 4. Mechanistic Interpretability (`mechanistic_interpretability.ipynb`)

### Papers
- **"A Mathematical Framework for Transformer Circuits"** (Elhage et al., 2021)
  - Foundational work in mechanistic interpretability for transformers
  - [Paper Link](https://transformer-circuits.pub/2021/framework/index.html)

- **"Locating and Editing Factual Associations in GPT"** (Meng et al., 2022)
  - Understanding how neural networks store and process factual information
  - [Paper Link](https://arxiv.org/abs/2202.05262)

- **"Towards Automated Circuit Discovery for Mechanistic Interpretability"** (Conmy et al., 2023)
  - Automated methods for finding computational circuits in neural networks
  - [Paper Link](https://arxiv.org/abs/2304.14997)

### Blogs & Resources
- **Anthropic Blog: "Transformer Circuits Thread"**
  - Series of posts on understanding transformer internals
  - [Blog Link](https://transformer-circuits.pub/)

- **Neel Nanda's Blog: "A Comprehensive Mechanistic Interpretability Explainer"**
  - Detailed introduction to mechanistic interpretability concepts
  - [Blog Link](https://www.neelnanda.io/mechanistic-interpretability/glossary)

- **Chris Olah's Blog: "Neural Networks, Manifolds, and Topology"**
  - Deep dive into the mathematical foundations of neural network interpretability
  - [Blog Link](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)

## 5. Causal Tracing & Feature Directions (`causal_tracing_directions.ipynb`)

### Papers
- **"Causal Tracing for Neural Network Interpretability"** (Meng et al., 2022)
  - Systematic method for tracing information flow through neural networks
  - [Paper Link](https://arxiv.org/abs/2211.02892)

- **"Discovering Latent Concepts Learned in BERT"** (Mu & Andreas, 2020)
  - Methods for finding meaningful directions in neural network representations
  - [Paper Link](https://arxiv.org/abs/2010.03550)

- **"Linear Algebra in Neural Networks"** (Elhage et al., 2022)
  - Understanding how neural networks perform linear transformations and feature mixing
  - [Paper Link](https://transformer-circuits.pub/2022/toy_model/index.html)

### Blogs & Resources
- **ROME Paper Website: "Locating and Editing Factual Associations in GPT"**
  - Interactive demonstrations of causal tracing in language models
  - [Website Link](https://rome.baulab.info/)

- **Anthropic Blog: "Superposition, Memorization, and Double Descent"**
  - Understanding how neural networks represent and manipulate information
  - [Blog Link](https://www.anthropic.com/index/superposition-memorization-and-double-descent)

- **David Bau's Lab: "Dissecting Neural Networks"**
  - Tools and techniques for understanding neural network internals
  - [Lab Website](https://baulab.info/)

## Miscellaneous References

### General Interpretability
- **"Interpretable Machine Learning"** (Christoph Molnar, 2020)
  - Comprehensive book covering interpretability techniques across ML
  - [Book Link](https://christophm.github.io/interpretable-ml-book/)

- **"Explainable AI: A Brief Survey on History, Research Areas, Approaches and Challenges"** (Adadi & Berrada, 2018)
  - Survey of the entire explainable AI field
  - [Paper Link](https://arxiv.org/abs/1909.10982)

### Tools & Libraries
- **Captum (PyTorch)**
  - Model interpretability library for PyTorch
  - [GitHub](https://github.com/pytorch/captum)

- **TransformerLens**
  - Library for mechanistic interpretability of transformers
  - [GitHub](https://github.com/neelnanda-io/TransformerLens)

- **Lucent**
  - Neural network visualization library
  - [GitHub](https://github.com/greentfrapp/lucent)