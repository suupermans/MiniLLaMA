# MiniLLaMA — 从零复现并现代化改造 GPT 架构

基于 nanoGPT 框架，从零实现字符级 GPT 的完整训练与生成管线，随后将架构现代化为
LLaMA 风格（RoPE / RMSNorm / SwiGLU），并实现 KV Cache 推理优化与消融实验。

> 10.6M 参数 · 字符级词表 (65) · Shakespeare 数据集 (1M 字符) · 单卡 RTX 3060 训练约 5 分钟

## 架构改造对照

| 组件 | GPT-2 原版 | mini-LLaMA 改造 | 动机 |
|---|---|---|---|
| 位置编码 | wpe 可学习查表（硬上限 256） | RoPE 旋转位置编码（零参数，天然相对位置） | 相对结构 + 可外推 |
| 归一化 | LayerNorm（减均值+bias） | RMSNorm（仅均方根缩放） | 更少计算，效果持平 |
| 前馈层 | GELU-MLP (4d) | SwiGLU 门控 FFN (8/3·d) | 门控表达力，参数量持平 8d² |
| 推理 | 每步全量 forward（O(N²) 冗余） | KV Cache（prefill/decode 分离，O(N)） | 消除历史重算 |

## 实验结果

### 1. 架构消融（同参数预算，各训 5000 步）

| 模型 | 最优 val loss | 达到步数 | 说明 |
|---|---|---|---|
| GPT-2 基线 (wpe/LN/GELU) | 【填：1.4719】 | 【填：5000】 | 收敛慢但未过拟合 |
| mini-LLaMA (RoPE/RMS/SwiGLU) | 【填：1.4574】 | 【填：2000】 | 收敛提速 2.5x；1M 字符数据下于 step 2500 后进入过拟合（train/val 分叉）|

### 2. KV Cache 推理加速

| 生成长度 | 朴素版耗时 | KV Cache 耗时 | 加速比 |
|---|---|---|---|
| 【填：100】 | 【填】 | 【填】 | 【填】 |
| 【填：250】 | 【填】 | 【填】 | 【填】 |

固定随机种子下，KV Cache 版输出与朴素版**逐字一致**（优化不改变语义）。

### 3. KV Cache 显存账（理论值）

`2(K/V) × 6层 × 1(batch) × 6头 × 255(token) × 64(维) × 4B ≈ 【填：4.7】MB`

## 运行

```bash
python data/shakespeare_char/prepare.py                                   # 数据准备
python train.py config/train_shakespeare_char.py --device=cuda --compile=False --batch_size=32
python sample.py --out_dir=out-shakespeare-char --num_samples=1 --temperature=0.8
```

## 目录结构

- `model.py` — 模型定义（mini-LLaMA 架构 + KV Cache）
- `train.py` — 训练循环（warmup/余弦衰减/梯度裁剪/混合精度）
- `sample.py` — 采样生成入口

> 本仓库 fork 自 karpathy/nanoGPT，架构与推理部分为个人改造。
