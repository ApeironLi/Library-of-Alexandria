#code_analyze #Transformer

- Note-Author: ApeironLi
- Version: 2023.7.23-1.0

## This is the code analyze of [[Note-vaswani2017attention]]. [代码来源](https://github.com/jadore801120/attention-is-all-you-need-pytorch)

Transformer的原始代码经过ApeironLi注释如下，用于借鉴其编写方式，只需要阅读train.py和Transformer文件夹中的内容即可。本文提到的“作者”，无特殊指代时均指Transformer的Pytorch版源码作者Yu-Hsiang Huang。

请注意，Pytorch已经对Transformer的实现提供给了很好的封装，本文是对Transformer源码的解析，如需搭建Transformer请参考代码分析：[[啥时候写一下]]

## Transformer 源码

---

### 1. 训练框架

Transformer将所有的数据处理都封装进了一个预处理函数，将数据处理和训练进行了隔离。
Transformer的训练代码(train.py)采用了经典的三层调用框架：

- main函数
- train函数
- train_epoch和eval_epoch函数

---

#### 1.1. 第一层Main函数

---

- 初始化参数：创建参数解析器parser来解析参数（以便于进行命令行传参），由于参数不多，作者并未单独将参数单独写成文件。
- 初始化训练设备：是否使用cuda，最简单的控制方式，较为复杂的方式可能还涉及指定训练用卡、多卡联动等等。
- 初始化网络：初始化网络对象（本处为1.2.部分的接入点）

---

- 初始化优化器：作者对优化器进行了**非常规**处理，将优化器放入网络文件夹Transformer中的Optim.py文件，并将训练中需要优化器做出的行为（主要是变更学习率等行为）都封装进class ScheduledOptim中，包括：
  - 类初始化：优化器、学习率、学习率控制参数等
  - 自动更新学习率的函数
  - 优化器基本行为：如.zero_grad，.step等

---

- 加载数据：作者对不同格式的输入数据分别编写了接口函数prepare_dataloaders和prepare_dataloaders_from_bpe_files。这些函数负责从制定路径抓入数据并直接传出一个dataloader。值得注意的是，由于NLP领域的数据集已经高度标准化，他们的dataset和datalaoder构造都直接调用了torchtext中的相关类（TranslationDataset和BucketIterator）。

---

- 调用下一层train函数：传入（网络、训练和测试data-loader、设备、优化器、参数）；不做返回（即该调用仅表示功能和逻辑上的层次关系）
- 复现性控制和提示信息打印（不重要）

---

#### 1.2. 第二层Train函数

---

- 训练记录和打印：
  - Tensorboard配置和使用
  - log_file：记录epoch数、损失、一些领域特殊指标和精度，有趣的是，作者在log_file中将这些信息以“表格”的形式记录，比较直观。
  - 打印配置：在Train函数中定义了一个函数print_performance将所有需要打印的中间提示信息疯装进入该函数（因为需要在train和val反复使用，封装后更直观精简）
- 训练-验证循环体：作者采用了训练验证一体化循环体而不是分写。
  - 调用下一层train_epoch函数：传入（模型、训练data-loader、优化器、参数）；返回该epoch的训练损失。
  - 调用下一层eval_epoch函数：传入（模型、测试data-loader、参数）；返回该epoch的测试损失。
  - 训练状态解析：解析当前训练状态（epoch、学习率、损失）并打印或记录。
  - 模型端点存储和加载：这是一个高度实用化的小功能，主要包括以下部分
    - 具备生成、加载和存储checkpoint的能力（即当前网络的训练状态）
    - 具备checkpoint最佳状态的解析能力（用于自动识别并存储网络的最佳训练装状态）
    - 具备根据checkpoint提供的信息调整网络状态的能力（用于在指定断点重启训练）

---

#### 1.3. 第三层Train_epcoh函数

---

- 误差计算和统计：深度神经网络通常需要多种误差的监督，需要分别计算这些误差并正确统计每个batch的误差值。
  - 正确计算误差并正确更新计数器

---

- Epoch内循环体：这是整个网络训练过程的核心，作者使用了tqdm构造迭代器，而非直接使用训练dataloader，这样做可以生成一个进度条，更直观地监视循环的执行进度。
  - 加载数据：从dataloader中解析需要的数据。
  - 前向传播：
    - 优化器.zero_grad
    - 模型预测
  - 反向传播：
    - 计算损失
    - loss.backward()
    - 优化器.step()
  - 计数器更新

---

### 2. Transformer网络架构代码

考虑到Transformer是一个复杂的网络架构（和他之前的一些网络相比确实算复杂了），作者将网络拆分为一些子模块放入了Transformer文件夹中，而不是将所有部分都放入model.py文件中。
与训练框架一样，Transformer的网络也具备高度的层次性，分为四个层次：

- Models.py
- Layers.py
- SubLayers.py
- Modules.py
  当然，Transformer将优化器也放入了网络文件夹中，这一点前文已经提过，不再赘述。我们也不必在意其他的几份文件，这些对于理解网络及其编程风格并无过多帮助。

---

#### 2.1. 第一层Models.py文件

作者在Models文件中规划了Transformer的顶层架构，即堆叠的编码-解码器架构当然还有PositionalEncoding层，不过该层的重要性并不如网络的主干结构。作者同样也将解码器所需的输入掩模功能分别通过两个函数get_pad_mask和get_subsequent_mask进行了封装。

- 位置编码层：本层不做过多解析，因该层为NLP问题专属。不过作者使用了一种有趣的编程技巧，在网络中初始化一个register_buffer对象，相当于在初始化网络时同时注册了一个buffer，这个buffer可以被PyTorch的优化器所识别和管理。这个buffer可以被视为模型中的固定参数，但是它不会被当做需要训练的参数来更新。在模型的前向传播过程中，这个buffer可以被用来存储一些需要在多个前向传播过程中共享的数据，比如BN层中的均值和方差。除此之外，使用self.register_buffer('xx', value)还可以保证这个buffer在模型保存和加载时能够正确地被保存和加载，从而保证了模型的可持续性。
- Encoder：编码器类的主要作用是将N个编码层堆叠在一起，并完成一些调整（如dropout、层归一化等），我们可以从成员对象和forward函数两个方面分析编码器对象：

  - 成员对象：
    - 输入嵌入层：嵌入层是NLP领域特有的层，旨在将属于离散空间的变量映射到连续空间中的编码向量以供网络进行后续处理。该层在其他大部分问题中都可以被一个简单的FC层直接替代。
    - 位置编码层：仅在第一个编码层之前使用（与原文一致）。
    - Dropout层：仅在第一个编码层之前使用（与原文一致）。
    - 层正则化层：仅在第一个编码层之前使用（与原文一致）。
    - 堆叠编码层：作者使用了一种经典的多层组装堆叠代码范式，使用nn.ModuleList和[EncoderLayer(xxx) for _ in range(xx)] 的方式将多个编码层组装堆叠在了一起。这个部分调用了第二层Layers.py文件中编写的编码层。
  - Forward函数：
    - 输入序列预处理：输入序列经过嵌入、第一次dropout和层正则化。
    - 堆叠编码层：使用N个编码层逐次对预处理后的编码进行深层次编码。该编码层以上一层输出的编码作为输入，输出本层编码和该层的Self-Attention（可选）。
    - Forward函数返回最终获取的编码和每层的Self-Attention列表（可选）。
- Decoder：解码器类更加复杂，需要递归循环地处理输出序列（涉及相对复杂掩模操作）。且需要从Encoder处获得注意力以计算Cross-Attention。作者将所有这些处理都封装进了Decoder特有的DecoderLayer中，同时在Decoder对象上仅增加了一个dec_enc_attn和一个encoder_output。

  - 成员对象：
    - 输出嵌入层：因为Decoder的输入是上次递归处理的输出序列。
    - 位置解码层、Dropout层和层正则化层。
    - 堆叠解码层
  - Forward函数：
    - 输入序列预处理：按照文章的写法，Decoder的输入应该要进行掩模以屏蔽掉未来信息。但是作者将这个功能都封装进了DecoderLayer，不得不说非常的优雅。
    - 堆叠解码层：与堆叠编码层不同，每一个解码层增加了几个输入，包括输入编码（用于计算Cross-Attention）、自注意力Mask（用于掩模解码层输入）、互注意力Mask（用于掩模调整互注意力的计算，这部分与文章不完全一致）。每个输出编码层输出本层编码和该层的Self-Attention和Cross-Attention。
    - Forward函数返回最终获取的编码、每层的Self-Attention列表（可选）和Cross-Attention列表（可选）。
- 主体网络Transformer类：包含Encoder和Decoder子类，和一个输出层。同样也负责调用get_pad_mask函数和get_subsequent_mask函数生成输入（src）和输出（tar）序列的mask。除了连接Encoder和Decoder外，Transformer类最主要的作用便是制作Encoder和Decoder输入序列的mask。

---

#### 2.2. 第二层Layers.py文件

作者在Models.py中调用了Layers.py中的编码和解码层，其中：

- 编码层：包含一个自注意力层（slf_attn）和一个位置前向传播层（pos_ffn）。自注意力层的Q、K、V都来自上一层编码层的输出（或输入序列），并接受输入mask的掩模。
- 解码层：包含一个自注意力层（slf-attn）、一个互注意力层（enc_attn）和一个位置前向传播层（pos_ffn）。自注意力层的Q、K、V来自上一层解码层的输出（或输出序列），互注意力层的K、V来自最后一层编码层的输出，Q来自上一层解码层的输出。
  Layers中所有的层均为SubLayers.py文件中MultiHeadAttention和PositionwiseFeedForward类的实例。

---

#### 2.3. 第三层SubLayers.py文件

---

- 多头注意力类（MultiHeadAttention）：
  - 输入：Q、K、V和输入序列Mask；
  - 输出：编码q（注意这里作者复用了输入Q的变量名q来表示输出编码）、注意力矩阵attn。
  - 参数：多头注意力的头数n_head；Key的维度d_k；Value的维度d_v；模型的维度d_model；
- 多头注意力的计算过程：
  - 获取输入序列维度：sz_b（batch_size）、len_q（Q序列长度）、len_k（K序列长度）、len_v（V序列长度）；
  - 在计算Dot-product注意力前先使用FC层分别处理Q、K和V并将其维度调整至：sz_b, n_head, len_x, d_x，同时为mask广播出头数维度。
  - 调用Modules.py中的Dot-product注意力模块计算注意力。
  - 调整输出编码q的维度为：sz_b, len_q, n_head, d_v
  - Dropout、layer_norm和残差连接。

---

- 位置前向传播类（PositionwiseFeedForward）：就是个带残差、droupout和layernorm的全连接层。

---

#### 2.4. 第四层Modules.py文件

该文件中只有一个ScaledDotProductAttention类，该类按照[[Note-vaswani2017attention]]中相关公式计算注意力矩阵并得到output，注意该模块自带Mask功能，并包含一个Dropout层。
