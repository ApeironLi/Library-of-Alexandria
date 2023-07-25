#Paper_Note #Pre-training #LLM

- Note-Author: ApeironLi
- Version: 2023.7.23-1.0

 **Attention**: This paper focus on **pre-trained** models rather than prompt.

### 1. History of Modern AI Development

#### 1.1. Training and Dataset

- The problem of **Data Hungry**: Since deep neural networks usually have a large number of parameters, they are thus easy to overfit and have poor generalization ability without sufficient training data.
- The emergence of **Large Scale Dataset**
- The problem of **Limited Human-Annotated Data**
- **Transfer Learning** and **Self-Supervised Learning**:
  - Transfer Learning: Pre-train and fine-tune stage.
  - Self-Supervised Learning: Leverage intrinsic correlations in the text as supervision signals instead of human supervision. (e.g. Masked out the last word and require the model to predict.)

#### 1.2. Model Architecture

- The problem of gradients vanishing or exploding lead to shallow models in early stages of NLP community.
- Transformers allow us to design large models with bigger size and more layers.

#### 1.3. Recent Trend

Unlabeled pre-training dataset + Self-supervised Learning + Transformer

- Pre-training Method: From transfer learning to self-supervised learning.
- Pre-training Dataset: From labeled dataset to unlabeled dataset.
- Model Architecture: Transformers allow sequential modeling to be deeper.

### 2. Transformer: A Powerful Tool

Transformer-Introduce-[[Note-vaswani2017attention]]
Transformer-[[Code-analyze-vaswani2017attention]]

本文后面的内容价值不大，虽然提到了GPT和BERT但是其实解释的不是很到位，可以不必阅读后续内容。
