"""
在预训练好的 mini-LLaMA 上做 LoRA SFT：
- 数据：从莎剧原文挖台词当回答，按关键词配模板问题，构造"Q:...\\nA:..."格式
- 损失：和预训练同一个 F.cross_entropy，只是回答部分才算 loss（指令部分 label=-1 屏蔽）
- 训练：只训 LoRA 的 A/B（apply_lora 已冻结其余参数）
- 产出：merge 后存到 out-sft/，sample.py 不改一行即可加载对比
"""
import os, pickle, random
import torch
import torch.nn.functional as F
from model import GPT, GPTConfig

# ---------------- 配置 ----------------
PRETRAINED_DIR = 'out-shakespeare-char'   # 预训练模型所在目录
OUT_DIR        = 'out-sft'                # SFT 产物目录
N_EXAMPLES     = 80
EPOCHS         = 16 #5 
LR             = 1e-3
BLOCK          = 256

random.seed(1337)
torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- 1. 加载词表 ----------------
meta = pickle.load(open('data/shakespeare_char/meta.pkl', 'rb'))
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]

# ---------------- 2. 构造 SFT 数据 ----------------
# 从原文里挖"像台词的行"当回答，按行内关键词配一个问题
text = open('data/shakespeare_char/input.txt', encoding='utf-8').read()
QUESTIONS = [  # (关键词, 模板问题)
    ("love",    "Tell me of love."),
    ("heart",   "What wisdom lies within the heart?"),
    ("king",    "Speak, what manner of king is worthy?"),
    ("death",   "What think you of death, good sir?"),
    ("night",   "How fares the night?"),
    ("father",  "What duty owes a son to his father?"),
    ("blood",   "Speak of blood and honour."),
    ("good",    "What is good, and what is noble?"),
    ("sir",     "Give counsel, sir."),
    ("lord",    "What would you of my lord?"),
    ("true",    "What is true, and what is false?"),
    ("fear",    "Why do men fear the dark?"),
    ("fortune", "Is fortune kind to brave men?"),
    ("soul",    "Whither goes the soul?"),
    ("world",   "What is this world to thee?"),
    ("time",    "What is time to the weary?"),
]

raw_lines = [l.strip() for l in text.split('\n')]
candidates = [l for l in raw_lines
              if 40 <= len(l) <= 120 and l[0].isupper() and l[-1] in '.?!']
random.shuffle(candidates)

examples, used = [], set()
for line in candidates:
    if len(examples) >= N_EXAMPLES or line in used:   # ← 答案行去重
        continue
    low = line.lower()
    for kw, q in QUESTIONS:
        if kw in low:
            try:
                encode(f"Q:{q}\nA:{line}")
            except KeyError:
                break
            examples.append((q, line))
            used.add(line)                             # ← 记的是答案行
            break
print(f"SFT 数据：{len(examples)} 条")
for q, a in examples[:3]:
    print(f"  Q:{q}\n  A:{a}\n")

# ---------------- 3. 加载预训练模型 + 挂 LoRA ----------------
ckpt = torch.load(f'{PRETRAINED_DIR}/ckpt.pt', map_location=device, weights_only=False)
model = GPT(GPTConfig(**ckpt['model_args']))
model.load_state_dict(ckpt['model'])
model.apply_lora(r=8, alpha=16)
model.to(device)

def preview(q="Tell me of love."):
    """看模型当前对某个问题的续写（训练前后对比用）"""
    prompt = f"Q:{q}\nA:"
    idx = torch.tensor([encode(prompt)], device=device)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=60, temperature=0.7, top_k=20)
    return ''.join(itos[i] for i in out[0].tolist()[len(prompt):])

model.eval()
print("=== 训练前（基座模型，只学过续写，不认识 Q/A 格式）===")
print(repr(preview()))

# ---------------- 4. SFT 训练循环 ----------------
model.train()
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

for epoch in range(EPOCHS):
    random.shuffle(examples)
    total_loss, n = 0.0, 0
    for q, a in examples:
        prompt = f"Q:{q}\nA:"
        full = prompt + a
        ids = torch.tensor(encode(full), dtype=torch.long, device=device)
        labels = ids.clone()
        labels[:len(encode(prompt))] = -1          # 指令部分不算 loss（ignore_index=-1）

        # 错位一格：位置 t 的输入预测 t+1 的 token（预训练同款规矩）
        logits, loss = model(ids[:-1].unsqueeze(0), targets=labels[1:].unsqueeze(0))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item(); n += 1

    print(f"epoch {epoch+1}/{EPOCHS} | train loss {total_loss/n:.4f}")
    model.eval()
    print("  样例续写:", repr(preview()))
    model.train()

# ---------------- 5. 合并 LoRA + 保存 ----------------
model.eval()
model.merge_lora()
os.makedirs(OUT_DIR, exist_ok=True)
torch.save({'model': model.state_dict(), 'model_args': ckpt['model_args'],
            'config': ckpt.get('config')},   # ★ 必须带上：sample.py 靠 config 里的 dataset 名找到 meta.pkl
           f'{OUT_DIR}/ckpt.pt')
print(f"\nSFT 完成，最优权重已合并保存至 {OUT_DIR}/ckpt.pt")
print("对比验收：")
print(f"  python sample.py --out_dir={OUT_DIR} --start=\"Q:Tell me of love.\\nA:\" --num_samples=1 --max_new_tokens=100 --temperature=0.7")
print("  再跑一次 --out_dir=out-baseline-wpe 或 PRETRAINED_DIR 的模型，对比格式差异")
