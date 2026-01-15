## 第六章 知识蒸馏（Knowledge Distillation）

在完成 LoRA 微调以及强化学习（DPO）之后，模型已经在参数效率、指令遵循和人类偏好对齐等方面得到了较为充分的优化。但由于模型规模本身受限，其整体表征能力和分布建模能力仍然受到上限约束。知识蒸馏正是在这一阶段引入，其目标不是重新塑造模型行为，而是利用一个更强的教师模型，将其在 token 级别学到的概率分布信息迁移到当前学生模型中，从而在不显著增加参数量的前提下进一步提升模型质量。
在 MiniMind 的训练流程中，知识蒸馏默认基于已经完成 SFT、LoRA 或 DPO 的权重继续训练。学生模型通常是当前可部署的小模型，而教师模型则是结构相同但隐藏维度更大、层数更多的模型，并且教师模型同样来自已经收敛的权重。蒸馏阶段不会再引入新的偏好信号，其训练数据可以直接复用 SFT 阶段的数据，这样可以避免额外的数据构造成本，同时保证分布一致性。
蒸馏训练中最关键的代码是蒸馏损失的定义。这里使用的是基于 KL Divergence 的经典蒸馏方法，教师模型的 logits 会先经过 temperature 缩放并转换为概率分布，学生模型则对同样缩放后的 logits 计算 log 概率，二者之间通过 KL 散度进行约束。temperature 的作用在于软化教师模型的输出分布，使得低概率 token 中隐含的信息能够被学生模型感知；最终的损失会乘以 temperature 的平方，以保证梯度尺度在不同温度下保持一致。

```python
def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    return (temperature ** 2) * kl
```
在单个训练 step 中，学生模型和教师模型会对同一输入序列同时进行前向传播。教师模型始终处于 eval 模式并完全冻结，仅用于生成 logits，不参与反向传播。为了适配不同规模模型，教师模型输出的 vocab 维度会被裁剪到与学生模型一致，避免因词表不对齐导致的无效梯度。真实标签的交叉熵损失与蒸馏损失会同时计算，并通过 alpha 参数进行加权融合，从而在“对齐真实标签”和“模仿教师分布”之间取得平衡。

```python

# 1) Ground-Truth CE Loss
ce_loss = F.cross_entropy(
    student_logits.view(-1, student_logits.size(-1)),
    Y.view(-1),
    ignore_index=0,
    reduction='none'
)
ce_loss = torch.sum(ce_loss * loss_mask_flat) / loss_mask_flat.sum()

# 2) Distillation Loss
distill_loss = distillation_loss(
    student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
    teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
    temperature=temperature
)

# 3) 总损失
loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps
```
从整体训练结构来看，知识蒸馏在工程实现上与强化学习阶段保持了高度一致。同样支持混合精度训练、梯度累积、梯度裁剪以及 DDP 分布式训练，并且可以从已有 checkpoint 自动恢复状态。这使得蒸馏可以被无缝插入到现有训练流水线中，而不需要额外改动数据加载或模型封装逻辑。

在实际运行时，蒸馏训练通常采用较小的学习率，以避免对前一阶段已经学到的能力造成破坏。alpha 一般设置在 0.3 到 0.7 之间，用于控制监督信号与蒸馏信号的相对权重；temperature 推荐在 1.0 到 2.0 范围内调节。教师模型的规模应明显大于学生模型，否则蒸馏带来的增益会非常有限。

下面给出一个典型的知识蒸馏训练启动示例，其风格和参数组织方式与前文 SFT 与强化学习阶段保持一致，便于统一管理实验配置。

```
nohup python train_distillation.py \
    --epochs 6 \
    --batch_size 32 \
    --learning_rate 5e-6 \
    --device cuda:0 \
    --accumulation_steps 1 \
    --log_interval 100 \
    --save_interval 100 \
    --max_seq_len 512 \
    --data_path ../dataset/pretrain_hq.jsonl \
    --student_hidden_size 512 \
    --student_num_layers 8 \
    --teacher_hidden_size 2048 \
    --teacher_num_layers 20 \
    --from_teacher_weight pretrain \
    --alpha 0.5 \
    --temperature 1.5 \
    --use_wandb \
    > train_distill.log 2>&1 &
```
from_teacher_weight 指向的是已经完成SFT、LoRA或DPO 的教师模型权重，这是蒸馏中唯一需要加载的权重文件，我们这里选用我们预训练的模型作为教师模型，这样让学生模型学习到我们已经有的教师模型的相关知识。student_hidden_size 和 student_num_layers 仅定义学生模型结构，源代码中不会尝试加载学生权重，因此学生模型会自动随机初始化。<br>
在完整训练范式中，知识蒸馏通常位于 LoRA 微调和强化学习之后，用于在不增加推理成本的前提下进一步压缩和整合模型能力。一个较为稳健的推荐流程是：预训练获得基础语言建模能力，SFT 对齐指令格式，LoRA 或全参微调强化特定任务能力，随后通过 DPO 等强化学习方法进行偏好对齐，最后使用知识蒸馏将大模型或强模型的分布知识迁移到最终可部署的小模型中。这种分阶段、逐步收敛的训练策略在算力受限的小模型场景下表现尤为稳定。