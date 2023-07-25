#Paper_Note #Transformer

- Note-Author: ApeironLi
- Version: 2023.7.23-1.0

## [[Code-analyze-vaswani2017attention]]

### 1. Connection between Transformer and Retrieval System

The key/value/query concept is analogous to retrieval systems.
For example, when you search for videos on Youtube, the search engine will map your **query** (text in the search bar) against a set of **keys** (video title, description, etc.) associated with candidate videos in their database, then present you the best matched videos (**values**). When you want to restrict which key-value pairs each query vector can attend, use Attention mask.

### 2. Basic Layers

- Main Hidden Vectors
  - Query set $\mathcal{Q}=\{q_1,...,q_n\}\in \mathbb{R}^{n\times d_k}$
  - Key set $\mathcal{K}=\{k_1,...,k_m\}\in\mathbb{R}^{m\times d_k}$
  - Value set $\mathcal{V}=\{v_1,...,v_m\}\in \mathbb{R}^{m\times d_v}$
- Scaled dot-product attention layer (Vanilla attention layer)
  - Layer hidden output $\mathcal{H}=\{h_1,...,h_n\}=\rm{ATT}(\mathcal{Q},\mathcal{K},\mathcal{V})=\mathcal{A}\mathcal{V}$
  - Attention operation $A=\rm{Softmax}(\rm{ATT-Mask}(\frac{\mathcal{Q}\mathcal{K}^T}{\sqrt{d_k}}))$. Dot-product operation is essentially used to measure the weight $a_{ij}$ to indicate how attended the query vector $a_i$ against the key vector $k_j$. # 相当于检索操作
  - Attention mask $\mathrm{ATT-Mask}(x)=-\infty$ if we do not want $q_i$ to attend $k_j$, otherwise $\mathrm{ATT-Mask}(x)=x$.
- Position-Wise Feed-Forward Layer: just a simple FC layer.
- Residual Connection and Normalization: Transformer applies residual connection and layer normalization between various neural layers, making the architecture of Transformer possible to be deep: $\mathcal{H}=\mathrm{A\&N}(X)=\mathrm{LayerNorm}(f(X)+X)$. $\mathrm{LayerNorm}(.)$ denotes the layer normalization operation.

### 3. Multi-head Attention Layer

多头注意力相当于将多个点乘注意力机制组装起来，每个点乘注意力机制负责提取一种语义特征（且具备自己的特征空间），组成一个高维特征向量。

- Layer hidden output

$$
\begin{align*}
\mathcal{H}&=\mathrm{MH-ATT}(\mathcal{Q},\mathcal{K},\mathcal{V})\\
&=\mathrm{Concat}(\mathcal{H}_1,...,\mathcal{H}_h)\mathcal{W}^O \\
\mathcal{H}_i&=\mathrm{ATT}(\mathcal{Q}\mathcal{W}_i^\mathcal{Q},\mathcal{K}\mathcal{W}_i^\mathcal{K},\mathcal{V}\mathcal{W}_i^\mathcal{V})\\
&= \mathcal{A}\mathcal{V}\mathcal{W}_i^\mathcal{V}
\end{align*}
$$

Among which $\mathcal{W}_i^\mathcal{Q},\mathcal{W}_i^\mathcal{K},\mathcal{W}_i^\mathcal{V}$ are respectively used to project the input $\mathcal{Q},\mathcal{K},\mathcal{V}$ into the feature space of the $i-th$ head attention. $\mathcal{H}_i$ represent the $i-th$ head attention which is used to capture the $i-th$ linguist feature.

### 4. Transformer

Transformer是一种新型Encoder-Decoder架构，包含N个堆叠的Encoder和Decoder。每个Encoder包含一个Self-Attention模块、一个Position-wise FC layer和一个残差和层正则化模层。每个Decoder则包含一个Masked Self-Attention模块、一个Cross-Attention模块和前馈及残差模块。
在工作时，Transformer的第一个Encoder接受Input Sequence并将其编码为表征，将表征输入下一个Encoder直至最后一个Encoder。
Transformer的第一个Decoder将上一次循环产生的字符添加在自己的输入序列上（这个输入序列即最终的输出序列，如果是第一次循环，则不需要添加字符）同时生成表征传递给下一个Decoder。同时所有Decoder的Cross-Attention层都接受来自最后一个Encoder的输出作为K和V。最后一个Decoder每次工作只预测一个（或一组）字符，在下一次工作时，该字符会作为输入加入Decoder的输入序列。

#### 4.1. Self-Attention

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.

In the encoding phase, given a word, the self-attention computes its attention scores by comparing it with all words in the input sequence. And such attention scores indicate how much each of the other words should contribute to the next representation of the given word.

#### 4.2. Masked Self-Attention

这个模块主要用于将Decoder输入序列（即最终的输出序列）中未来的信息去除，因为在生成时未来的信息是未知的（即Decoder输入序列中尚未预测的部分是随机值/零值）
Whose attention matrix satisfies $\mathcal{A}_{ij}=0,i>j$. In the decoding phase, the masked self-attention only decodes one representation from left to right at one time. Since each step of the decoding phase only consults the previously decoded results, they thus require to add the masking function into the self-attention.

#### 4.3. Cross-Attention (Encoder-Decoder Attention)

The queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models.
