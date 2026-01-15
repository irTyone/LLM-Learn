## 第五章 强化学习对齐训练（RL Alignment）

在大模型完成预训练与监督微调之后，模型已经具备较强的语言建模能力，但在输出偏好、人类价值对齐、安全性与稳定性方面仍存在不足。因此，通常需要在 SFT 之后引入强化学习对齐阶段，通过偏好数据或奖励信号进一步优化模型输出。目前主流的大模型强化对齐方法主要包括 DPO、PPO、GPO 以及 GRPO，它们在目标函数形式、工程复杂度和资源消耗上各有不同。

DPO（Direct Preference Optimization）是一种直接基于偏好对进行优化的方法，不再显式训练奖励模型，也不使用传统强化学习中的策略采样机制。设在同一输入 x 下，y⁺ 为偏好回答，y⁻ 为非偏好回答，π_θ 表示当前策略模型，π_ref 表示冻结的参考模型，则 DPO 的优化目标为：

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathcal{L}_{\text{DPO}}=-\mathbb{E}_{(x,y^+,y^-)}\left[\log\sigma\left(\beta\left(\log\pi_\theta(y^+|x)-\log\pi_\theta(y^-|x)-\left(\log\pi_{\text{ref}}(y^+|x)-\log\pi_{\text{ref}}(y^-|x)\right)\right)\right)\right]" />
</p>

其中 σ(·) 为 Sigmoid 函数，β 为温度系数，用于控制策略模型偏离参考模型的程度。该目标函数鼓励模型在相对概率上更偏向优选回答，同时通过参考模型项约束更新幅度，从而保证训练稳定性。

PPO（Proximal Policy Optimization）是 RLHF 体系中最经典的策略优化算法，其核心思想是在最大化期望奖励的同时，通过裁剪或 KL 约束限制策略更新幅度。PPO 的核心目标函数为：

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathcal{L}_{\text{PPO}}=\mathbb{E}_t\left[\min\left(r_t(\theta)A_t,\;\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\right)\right]" />
</p>

其中

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}" />
</p>

A_t 为优势函数，ε 为裁剪阈值。在语言模型对齐任务中，通常还会加入 KL 约束项：

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathcal{L}=\mathcal{L}_{\text{PPO}}-\lambda\,\text{KL}(\pi_\theta\|\pi_{\text{ref}})" />
</p>

PPO 需要同时维护策略模型、参考模型和奖励模型，并进行在线采样，因此显存与算力开销显著高于 DPO。

GPO（Generalized / Grouped Policy Optimization）是 PPO 的一种泛化形式，其核心思想是通过分组或聚合多个样本来降低奖励噪声。在 GPO 中，优势函数由组级别统计量估计，其目标函数可表示为：

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathcal{L}_{\text{GPO}}=\mathbb{E}_g\left[\mathbb{E}_{t\in g}\left[\min\left(r_tA_g,\;\text{clip}(r_t,1-\epsilon,1+\epsilon)A_g\right)\right]\right]" />
</p>

其中 g 表示一个样本组，A_g 为组级别优势估计。GPO 在形式上仍属于 PPO 框架，因此依然依赖奖励模型和 KL 约束，其资源消耗与 PPO 接近，但在噪声较大的偏好数据场景中更稳定。

GRPO（Group Relative Policy Optimization）是一种不依赖显式奖励模型的相对策略优化方法。设在同一输入 x 下采样得到 n 个候选回答 {y₁,…,yₙ}，GRPO 通过组内相对对数概率构造隐式优势，其目标函数可写为：

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathcal{L}_{\text{GRPO}}=-\mathbb{E}\left[\sum_{i=1}^{n}\log\pi_\theta(y_i|x)\left(\log\pi_\theta(y_i|x)-\frac{1}{n}\sum_{j=1}^{n}\log\pi_\theta(y_j|x)\right)\right]" />
</p>

该方法通过组内相对概率差异进行优化，使高概率输出得到强化、低概率输出被抑制。GRPO 在思想上介于 DPO 与 PPO 之间，不需要奖励模型，工程复杂度和显存开销低于 PPO、高于 DPO。

综合来看，DPO 与 GRPO 更适合中小模型或资源受限场景，PPO 与 GPO 则在高质量人工反馈和大规模算力条件下具有

### 代码实现
以下以train_dpo.py作为例子运行。<br>
在代码实现上，DPO 需要同时维护两个模型，一个是参与训练的策略模型，另一个是完全冻结的参考模型。策略模型在加载完成监督微调权重后，会通过 apply_lora 接口注入 LoRA Identity 结构，并仅对名称中包含 lora 的参数开启梯度，其余参数全部冻结；参考模型使用相同的基础权重初始化，但不注入 LoRA，也不参与反向传播，仅用于提供稳定的概率基准。相关核心代码如下所示：

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
apply_lora(model)
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
ref_model.eval()
ref_model.requires_grad_(False)
```
DPO 的训练数据由偏好对组成，每一条样本都包含同一指令下的 chosen 与 rejected 两种回答。在训练过程中，这两类样本会在 batch 维度上拼接后送入模型进行前向计算。通过对 logits 执行 log_softmax 并结合真实标签索引，可以得到逐 token 的对数概率分布，再在 mask 的约束下计算序列级别的平均对数概率，用于后续偏好对比。
DPO 损失函数的核心思想并不是最大化绝对奖励，而是让策略模型在 chosen 与 rejected 回答之间的概率差，相对于参考模型的概率差更大。具体实现中，首先分别计算策略模型和参考模型在 chosen 与 rejected 回答上的对数概率差值，然后将两者相减，并通过带温度系数 beta 的 logsigmoid 函数进行优化，这种损失设计保证了策略模型的更新始终受到参考模型的约束，使模型在偏好对齐过程中不会发生过度漂移。由于仅有 LoRA Identity 参数参与反向传播，梯度图规模非常小，整体显存占用相较于预训练阶段仅有轻微增加，通常远低于全参数 DPO 或 PPO 的需求。

在实际运行时，可以通过编写独立的训练脚本来启动基于 LoRA Identity 的 DPO 训练流程，例如：
```
nohup python train_dpo.py \
    --epochs 20 \
    --hidden_size 2048 \
    --device "cuda:6"\
    --batch_size 16 \
    --learning_rate 1e-4 \
    --accumulation_steps 8 \
    --log_interval 10 \
    --save_interval 100 \
    --data_path ../dataset/dpo.jsonl \
    --num_hidden_layers 20 \
    --use_wandb \
     --from_resume 1\
    --from_weight lora_identity \
    > train_lora.log 2>&1 &

```

上述运行命令中，from_weight 指向已经完成监督微调的权重作为对齐起点，save_weight 用于标识本次训练得到的 LoRA Identity 强化学习权重，训练过程中仅保存和更新 LoRA 相关参数。通过这种方式，可以在保持模型通用能力稳定的前提下，高效完成偏好对齐训练，为后续推理部署或进一步对齐实验提供可靠基础。