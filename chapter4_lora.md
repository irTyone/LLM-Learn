## 第四章 LoRA 微调
### lora概念
LoRA（Low-Rank Adaptation）是一种参数高效微调方法，其核心思想是在不改变预训练模型原有参数的前提下，通过在模型中的部分线性层引入低秩可训练矩阵，使模型在特定任务或风格上获得额外的能力增强。与监督微调不同，LoRA 不会对模型的全部参数进行更新，而是仅对新增的低秩参数进行训练，因此在显存占用、训练时间以及模型稳定性方面具有明显优势，尤其适合算力受限的小模型或单卡训练场景。
在标准 Transformer 架构中，模型的大部分参数集中在注意力机制和前馈网络中的线性变换层。LoRA 的做法并不是直接对这些权重矩阵进行更新，而是为它们增加一个可学习的增量项，该增量由两个低秩矩阵相乘得到。原始权重在整个训练过程中保持冻结状态.
LoRA 微调阶段所使用的数据在格式上与监督微调阶段保持一致，通常采用指令与回答配对的 jsonl 结构。这类数据的目标并不是让模型学习新的通用语言知识，而是引导模型在已有知识基础上，调整输出风格或增强某一类能力。因此在数据规模上，LoRA 通常远小于监督微调甚至预训练阶段的数据规模。在实际工程中，即使只有较小规模的数据集，也能够取得较为明显的效果提升。
在代码实现上，LoRA 微调首先需要在模型中注入 LoRA 结构。模型加载完成后，通过调用 apply_lora 函数对模型结构进行遍历，在指定的线性层中插入 LoRA 模块。该过程不会改变模型的前向计算逻辑，也不会影响原始参数的数值，仅为模型增加了新的可训练参数分支。

```python
from model.model_lora import apply_lora
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
apply_lora(model)
```
完成 LoRA 模块注入后，需要对模型参数进行冻结控制。通过遍历模型中的全部参数，并根据参数名称是否包含 lora 字样来判断其是否参与训练，可以确保只有 LoRA 模块中的参数会在反向传播过程中被更新，其余参数始终保持冻结状态。这种训练策略能够有效避免灾难性遗忘问题，同时显著降低显存占用，使得在较小显卡上训练较大模型成为可能。
```python
lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True
        lora_params.append(param)
    else:
        param.requires_grad = False       
```

在优化器的设置上，只需要将 LoRA 参数传入优化器即可。这样在整个训练过程中，优化器的状态维护与梯度更新都仅作用于 LoRA 参数，不仅减少了显存开销，也降低了训练过程中的数值不稳定风险。

```python
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
```
完成上述配置后即可启动 LoRA 微调训练。与监督微调相比，LoRA 阶段通常使用更小的学习率和更少的训练轮次即可达到稳定效果。训练过程中模型会按照设定的保存间隔周期性保存 LoRA 权重文件，用于后续推理或能力加载。
### 代码修改
在原代码的model.py中，使用了原作者的lora代码程序，但会有因为张量形状的不适配导致程序报错，浏览了minimind相关交流的帖子后没有发现有别人提到这个问题，所以我哦在这里修改了作者的源代码，修改了之后正常运行，如果你不想在运行时遇到我的问题，可以如第一章所示克隆我fork后的仓库。以下是我修改在model.py中修改的方法：
```
def apply_lora(model, rank=8):
    # 注意：MiniMind 建议给 q_proj, k_proj, v_proj, o_proj 等添加 LoRA
    names_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 如果只想针对某些层，可以加过滤条件，如：
            # if any(key in name for key in ["q_proj", "v_proj"]): 
            names_to_replace.append(name)

  
    for name in names_to_replace:
        
        parts = name.split('.')
        child_name = parts[-1]
        parent_name = '.'.join(parts[:-1])
        
        
        parent = model.get_submodule(parent_name)
        # 获取原始线性层
        original_layer = getattr(parent, child_name)
        

        new_module = LoRA(original_layer, rank).to(original_layer.weight.device)
        setattr(parent, child_name, new_module)
```
### 运行实例
```
nohup python train_lora.py \
    --epochs 20 \
    --hidden_size 2048 \
    --device "cuda:6"\
    --batch_size 16 \
    --learning_rate 1e-4 \
    --accumulation_steps 8 \
    --log_interval 10 \
    --save_interval 100 \
    --data_path ../dataset/lora_identity.jsonl \
    --num_hidden_layers 20 \
    --use_wandb \
     --from_resume 1\
    --from_weight full_sft \
    > train_lora.log 2>&1 &

```
LoRA 微调过程中保存的并不是完整模型参数，而是仅包含 LoRA 模块权重的独立文件。这类权重文件体积较小，便于存储与分发，同时可以在同一基础模型上加载不同的 LoRA 权重，实现多任务或多风格的快速切换。在完整训练流程中，通常先通过预训练与监督微调获得一个稳定的基础模型，再通过 LoRA 为模型注入特定领域或特定风格的能力，从而在训练成本与模型效果之间取得良好的平衡。

### lora可训练分析
|处理器|显卡|批次大小|显卡数|训练轮次|训练时间|参数量|显存占用|
|----------|------|---|------|------|---|---|---|
| Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz | NVIDIA A40 (46 GB) | 64 | 1 |20|2min|1B|8-10G|
| Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz | NVIDIA A40 (46 GB) | 64 | 1 |20|2min|272.418|3-4G|

lora训练所占用的显存与预训练和sft全量微调相比，占用资源显著下降，对于显存为4GB的个人显卡可以勉强训练0.2B的minimind模型，而8GB和16GB则能勉强运行1B左右的微调训练。注意，深度学习模型训练的过程所占显卡内存的大小是时刻变化的，勉强能够满足代表着显资源可能随时溢出导致训练失败。