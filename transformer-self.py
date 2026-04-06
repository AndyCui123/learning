# __coding__: UTF_8
# created by c00412291
import torch
import torch.nn as nn
import torch.nn.functional as F

# 超参数
d_model = 128    # 模型维度
n_heads = 8      # 注意力头数
d_k = d_model // n_heads  # 每个头的维度
d_ff = 256       # FFN 中间层维度
max_len = 100    # 最大序列长度

# -----------------------------------------------------------------------------
# 1. 位置编码 Positional Encoding
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, L, d_model]
        return x + self.pe[:, :x.size(1)]

# -----------------------------------------------------------------------------
# 2. 单头注意力（为了看懂，先写单头）
# -----------------------------------------------------------------------------
def scaled_dot_product_attention(Q, K, V):
    # Q/K/V: [B, n_heads, L, d_k]
    attn_score = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attn_weight = F.softmax(attn_score, dim=-1)
    output = torch.matmul(attn_weight, V)
    return output, attn_weight

# -----------------------------------------------------------------------------
# 3. 多头注意力 Multi-Head Attention
# -----------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        B, L = q.size(0), q.size(1)

        # 线性变换 + 拆多头
        Q = self.Wq(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        out, attn = scaled_dot_product_attention(Q, K, V)

        # 拼接多头
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.Wo(out)

# -----------------------------------------------------------------------------
# 4. Feed Forward 网络
# -----------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# -----------------------------------------------------------------------------
# 5. 单个 Transformer Block
# -----------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 注意力 + 残差 + 归一化
        x = self.norm1(x + self.attn(x, x, x))
        # FFN + 残差 + 归一化
        x = self.norm2(x + self.ffn(x))
        return x

# -----------------------------------------------------------------------------
# 6. 完整迷你 Transformer
# -----------------------------------------------------------------------------
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=8, d_ff=256, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # 词嵌入 + 位置编码
        x = self.emb(x)
        x = self.pe(x)

        # 过 N 层 Transformer Block
        for block in self.blocks:
            x = block(x)

        # 输出预测下一个 token
        return self.fc(x)

# -----------------------------------------------------------------------------
# 测试跑一遍
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    vocab_size = 1000  # 随便设一个词典大小
    model = MiniTransformer(vocab_size)

    # 构造一个假输入：batch=2, seq_len=10
    x = torch.randint(0, vocab_size, (2, 10))
    out = model(x)

    print("输入 shape:", x.shape)    # [2, 10]
    print("输出 shape:", out.shape)  # [2, 10, 1000]
