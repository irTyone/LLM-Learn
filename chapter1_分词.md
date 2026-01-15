# 大模型原理与实践
本教程在介绍大模型原理的同时，也会讲解如何书写代码与代码运行
## 环境准备
拉取仓库<br>
`git clone https://github.com/jingyaogong/minimind.git`<br>
随后创建自己的环境<br>
`conda create -n <你的环境名字>`<br>
启动环境<br>
`cd minimind`<br>
`conda activate <你的环境名字>`<br>
最后安装包<br>
`pip install -r requirements.txt`<br>
注意：不建议用过高版本的python,torch以及其他相关包对3.14或3.13版本的python还没有做到完全支持，建议使用3.10到3.11的包，比如<br>
`conda create -n <你的环境名字 python=3.11>`<br>
### 注意
原作者的训练代码显存占用异常大，有一定的不正常，其次，参数配置并没有完全以命令行格式进行，不仔细读源代码使用默认配置可能导致训练效果，因此自己fork了一版，一是训练占用资源优化、二是配置重写。我的仓库代码为：https://github.com/irTyone/LLM-Learn

##  第一章 分词
分词器是大模型预训练的初始步骤，大模型的训练的一般过程为将句子映射为相对应的词元id、词元id经过嵌入层映射为嵌入向量、并经过transformer层和前馈神经网络模块后，输出相应的结果。因此在给大模型训练前，需要先训练一个分词器
### BPE分词器
目前大模型训练普遍使用的分词器算法为BPE：BPE 算法的核心在于重复地将语料库中最常出现的相邻字节对合并成一个新的、更长的符号（Token）。
1. 输入： 一个大规模的文本语料库
2. 目标： 达到预定的词汇表大小（例如vocab_size=6400）。
3. 过程（训练）：
4. 初始化词汇表为所有单个字节（基础词汇）。
4. 扫描整个语料库，统计所有相邻符号对的出现频率。
4. 选择频率最高的相邻符号对，将其合并成一个新的符号，并添加到词汇表中。
重复步骤 2 和 3，直到达到目标词汇表大小。<br>
我们先打开分词器代码，先
`cd scripts`<br>
然后找到train_tokenizer.py文件，可以看到BPE分词器的定义代码
```
    #json数据文件地址
    data_path = '../dataset/pretrain_hq.jsonl'
    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # 定义特殊token
    special_tokens = ["<unk>", "<s>", "</s>"]
    # 设置训练器并添加特殊token
    trainer = trainers.BpeTrainer(
        vocab_size=6400,
        special_tokens=special_tokens,  # 确保这三个token被包含
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    # 读取文本数据
    texts = read_texts_from_jsonl(data_path)
    # 训练tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)
    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()
    # 检查特殊token的索引
    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("</s>") == 2
    # 保存tokenizer
    tokenizer_dir = "../model/minimind_tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("../model/minimind_tokenizer")
```
在以上代码中是设置了ByteLevel：<br>
`tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)`<br>
ByteLevel 预分词/解码器（ByteLevel Pre-tokenizer/Decoder）是 Hugging Face 的 tokenizers 库中实现的一种分词策略，它最初是为了配合 GPT-2、RoBERTa 等模型，解决传统 BPE 分词的局限性而设计的。在标准的 BPE 流程开始之前，预分词器会将原始文本进行第一次分解。BytLevel 的作用： 它会将输入文本中的每一个字符（包括空格和标点符号）映射到其对应的 UTF-8 字节表示，并将这些字节视为最基本的、不可再分的初始“符号”
#### 上下文感知
config这一数据结构是设置分词器相关参数的，其作用如下所示：
| 参数 | 值 | 说明 |
|------|----|------|
| tokenizer_class | PreTrainedTokenizerFast | 定义加载器：告诉 transformers 库在加载此分词器时应该使用哪个 Python 类。PreTrainedTokenizerFast 是加载基于 Rust 的高性能 tokenizers 库模型的标准类。 |
| model_max_length | 32768 | 上下文长度限制：定义模型能够处理的最大序列长度，用于在编码长文本时进行截断或填充。 |
| legacy | True | 兼容性标志：用于确保分词器在旧版或特定版本的 transformers 库中能正确运行。 |
| unk_token | "<unk>" | 未知 Token：定义用于替换无法识别字符的符号。ByteLevel BPE 通常不会生成它，但必须定义。 |
| bos_token | "<s>" | 句子开始 Token：标记序列开头的符号（Begin Of Sentence）。 |
| eos_token | "</s>" | 句子结束 Token：标记序列结束的符号（End Of Sentence）。 |
| pad_token | "<unk>" | 填充 Token：用于将不同长度的序列填充到相同长度，这里复用了 <unk>。 |
| additional_special_tokens | [] | 额外特殊 Token：用于存放除了上述核心 Token 之外的其他特殊符号，例如 <sep> 或 <mask>，本例中没有。 |
| add_bos_token | False | 自动添加 BOS：控制在调用编码方法时是否默认在输入序列开头添加 bos_token (<s>)，False 表示需要手动添加。 |
| add_eos_token | False | 自动添加 EOS：控制是否默认在输入序列结尾添加 eos_token (</s>)，False 表示由用户或模板控制。 |
| add_prefix_space | False | 前缀空格：控制是否在第一个 Token 前添加空格，与 ByteLevel 预分词器中的设置保持一致。 |
| clean_up_tokenization_spaces | False | 空格清理：控制解码时是否清理多余空格（例如句号前），False 保持字节级忠实还原。 |
| spaces_between_special_tokens | False | 特殊 Token 间的空格：解码时是否在特殊 Token 之间添加空格，False 确保紧密相连。 |
| added_tokens_decoder | {...} | 解码器映射：提供 ID 到 Token 的映射，用于特殊 Token。保证即使没有完整词汇表，这些特殊符号也能被正确识别。示例：ID 0: <unk>、1: <s>、2: </s>。 |
| chat_template | "{% if messages...}" | 最重要的作用：定义一个 Jinja 模板，用于 tokenizer.apply_chat_template() 方法将对话历史（messages 列表）转换为模型可读取的单行字符串。 |

chat_template 将整个对话历史（包括系统指令、所有用户消息和所有助手回复）按照规定的格式拼接在一起，然后将这个长字符串送入模型。<br>
初次使用： 模板只包含系统指令和第一条用户消息。<br>
在对话中（使用记忆）： 模板包含：[系统指令] + [第一轮用户问答] + [第二轮用户问答] + ... + [当前轮用户问题] + [等待模型回答的起始标记]。<br>
作者的定义如下的一个列表：
```
messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
```
#### 数据集下载
介绍完代码的具体参数之后，我们运行代码要从网站上进行下载，如下是作者给出的数据下载地址<br>
https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
![](数据下载页面.png)<br>
注意，下载的数据集一定要在项目结构中的dataset文件夹下，否则需要改变一下代码。
#### 运行
按照上面的操作后只需要直接运行代码即可：<br>
`python train_tokenizer.py`