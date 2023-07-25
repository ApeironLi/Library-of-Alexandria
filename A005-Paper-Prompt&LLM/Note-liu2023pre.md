#Paper_Note #Prompt #LLM 

---
### 1. Two Sea Changes of NLP Community
- Pre-train and Fine-tune Paradigm
	- Pre-train a language model (LM) on large dataset.
	- Fine-tune these LMs using task-specific objective functions.
	- Objective engineering is important!
- Pre-train, Prompt and Predict
	- Pre-train a LM on large dataset in unsupervised fashion.
	- Reformulate the downstream tasks to look more like those solved during original LM training with the help of a textual prompt. (Blank Filling Game!)
	- Few-shot or even Zero-shot Learning becomes possible.
	- Prompt Engineering is important!

---
### 2. Abstract of Prompting
#### 2.1. Terminology
Input $x$, Output $y$, Prompting Function $f_{prompt}(x)$, Prompt $x^{\prime}$, Filled Prompt $f_{fill}(x^{\prime},z)$, Answered Prompt $f_{fill}(x^{\prime},z^{*})$, Answer $z$.
#### 2.2. Three steps of Prompting
- Prompt Addition
Modify the input text $x$ into a prompt ${x^{\prime}=f_{prompt}(x)}$ by: First, apply a template(A string with an input slot [x] and answer slot [z]). Then fill slot [x] with $x$. Prompt addition can be classified into cloze prompt and prefix prompt.
- Answer Search
Search for the answer $\hat{z}$ with highest probability in the pre-trained LM by: First define the space of $z\in\mathbb{Z}$. Then search over $\mathbb{Z}$ by calculating the probability of filled prompts using a pre-trained LM $P(.;\theta)$:
$$
\hat{z}=\underset{z\in\mathbb{Z}}{search}P(f_{fill}(x^{\prime},z);\theta)
$$
This search function could be an argmax search that searches for the highest-scoring output, or sampling that randomly generates outputs following the probability distribution of the LM.
- Answer Mapping
Finally, we would like to go from the highest-scoring answer $z^{\prime}$ to the highest-scoring output $y^{\prime}$. (Only designed for specific tasks!)

---
### 3. Pre-training
Training Objective: Predicting the probability of text $x$.
- Standard Language Model (SLM): Predict $P(x)$ in an autoregressive fashion. (Suitable for prefix prompt.)
- Text Reconstruction (TR, CTR and FTR): Predict $P(x|\hat{x})$, $\hat{x}=f_{noise}(x)$. (Suitable for cloze prompt.) 
Note: Noise function is also critical for pre-training. (Mask, Replace, Delete, Permute, Rotate, Concatenation...)

---
### 4. Prompt Engineering
Prompt engineering aims to bridge the gap between upstream pre-training task and downstream tasks by creating a prompting function $f_{prompt}(x)$. The most important part of Prompt engineering is prompt template engineering, which includes **Prompt Shape Design** and **Prompt Template Design**.
#### 4.1. Prompts Shape
- Prefix prompt (Generation, L2R LMs)
- Cloze prompt (Masked LMs).
#### 4.2. Prompt Template Design Methods
Manual and Automated Template Engineering.
Static and Dynamic Prompt template.
##### 4.2.1. Discrete Prompts (Hard Prompts)
Automatically search for templates described in a discrete space.
Example: Prompt Mining, Prompt Paraphrasing, Gradient-based Search, Prompt Generation and Prompt Scoring.
##### 4.2.2. Continuous Prompts (Soft Prompts)
Prompting in the embedding space of the model.
(1) Relax the constraint that the embeddings of template words be the embeddings of natural language (e.g., English) words. (2) Remove the restriction that the template is parameterized by the pre-trained LM’s parameters.
Example: Prefix Tuning, Tuning Initialized with Discrete Prompts, Hard-Soft Prompt Hybrid Tuning.

---
### 5. Answer Engineering
**Prompt**: design appropriate inputs for prompting methods.
**Answer**: search for an answer space $Z$ and a map to the original output $Y$ that results in an effective predictive model.
#### 5.1. Answer Shape
- Tokens: One of the tokens in the pre-trained LM’s vocabulary, or a subset of the vocabulary.
- Span: A short multi-token span. (Cloze Prompts)
- Sentence: A sentence or document. (Prefix Prompts)
#### 5.2. Answer Space Design Methods
Design answer space and mapping function from answer to output.
Manual and Automated answer space design.
##### 5.2.1. Discrete Answer Search
Example: Answer Paraphrasing, Prune-then-Search, Label Decomposition.
##### 5.2.2. Continuous Answer Search
Example: Answer Paraphrasing, Prune-then-Search, Label Decomposition.

---
### 6. Multi-Prompt Learning
Constructing multiple prompts for a task is more effective than using a single prompt.
#### 6.1. Prompt Ensembling
Prompt ensembling is the process of using multiple unanswered prompts (e.g. different answer slots) for an input at inference time to make predictions.
#### 6.2. Prompt Augmentation
Prompt augmentation, also sometimes called demonstration learning, provides a few additional answered prompts that can be used to demonstrate how the LM should provide the answer to the actual prompt instantiated with the input $x$.
#### 6.3. Prompt Composition
For those composable tasks, which can be composed based on more fundamental subtasks, we can also perform prompt composition, using multiple sub-prompts, each for one subtask, and then defining a composite prompt based on those sub-prompts.
#### 6.4. Prompt Decomposition
Break down the holistic prompt into different sub-prompts, and then answer each sub-prompt separately.

---
### 7. Training Strategies for Prompting Methods
#### 7.1. Training Settings
- **Zero-Shot Learning**: simply take a LM that has been trained to predict the probability of text $P(x)$ and applying it as-is to fill the cloze or prefix prompts defined to specify the task.
- **Full-Data Learning**: a reasonably large number of training examples are used to train the model.
- **Few-Shot Learning**: a very small number of examples are used to train the model.
#### 7.2. Pre-trained & Prompts Parameters
Table-6 in this paper is quite intuitive.
- LM Params
- Prompt Params

---

