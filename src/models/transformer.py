import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F


def _mha_shape_check(query: Tensor, key: Tensor, value: Tensor) -> bool:
    """检查MHA各个输入Tensor的形状。

    Returns:
        bool: 输入是否batched。
    """
    is_batched = None
    if query.dim() == 3:
        is_batched = True
        assert key.dim() == 3 and value.dim == 3, (
            "对于 batched 3-D `query`，`key`和`value`应为3-D，"
            f"然而输入分别为{key.dim()}-D和{value.dim()}-D"
        )
        assert query.shape[0] == key.shape[0] == value.shape[0], (
            "对于 batched 3-D `query`、`key`和`value`，批次大小（batch_size）应该相等，"
            f"然而输入分别为{query.shape[0]}、{key.shape[0]}和{value.shape[0]}"
        )
    elif query.dim() == 2:
        is_batched = False
        assert key.dim() == 2 and value.dim() == 2, (
            "对于 unbatched 2-D `query`，`key`和`value`应为2-D，"
            f"然而输入分别为{key.dim()}-D和{value.dim()}-D"
        )
    else:
        raise AssertionError(
            f"`query`应为 unbatched 2-D tensor 或 batched 3-D tensor，然而输入为{query.dim()}-D"
        )

    assert query.shape[-1] == key.shape[-1], (
        "`query`和`key`要求同样的特征大小（embedding_size），"
        f"然而输入分别为{query.shape[-1]}和{key.shape[-1]}"
    )
    assert key.shape[-2] == value.shape[-2], (
        "`key`和`value`要求同样的序列长度（sequence_len），"
        f"然而输入分别为{key.shape[-2]}和{value.shape[-2]}"
    )

    return is_batched


def _scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout: nn.Dropout = None,
) -> tuple[Tensor, Tensor]:
    """计算attention"""
    d_k = query.shape[-1]
    # torch的矩阵乘法支持带batch的乘法，因此二维以上的矩阵也可以相乘
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        # mask == 0的位置都设置为负无穷
        scores = scores.masked_fill(mask == 0, float("-inf"))
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    attention_value = scores @ value
    return attention_value, scores


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_count):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_count = head_count
        self.q_weight = nn.Linear(embedding_dim, embedding_dim)
        self.k_weight = nn.Linear(embedding_dim, embedding_dim)
        self.v_weight = nn.Linear(embedding_dim, embedding_dim)
        # 输出权重矩阵W_O
        self.output_weight = nn.Linear(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, q_seq: Tensor, k_seq: Tensor, v_seq: Tensor, mask: Tensor | None = None
    ):
        r"""进行multi-head attention的计算"""
        is_batched = _mha_shape_check(q_seq, k_seq, v_seq)
        # 如果QKV是2维的，增加一个batch维度
        if is_batched is not True:
            q_seq = q_seq.unsqueeze(0)
            k_seq = k_seq.unsqueeze(0)
            v_seq = v_seq.unsqueeze(0)

        batch_size, src_seq_len, input_embedding_dim = q_seq.shape
        _, tgt_seq_len, _ = k_seq.shape
        # !这里为了简化模型，假定QKV的embedding_dim全部相等
        if input_embedding_dim != v_seq.shape[-1]:
            raise ValueError(
                "`value`的维度（embedding_dim）要求与`query`和`key`相等，"
                f"而输入的`value`维度是{v_seq.shape[-1]}，`query`和`key`的维度是{input_embedding_dim}"
            )
        if input_embedding_dim != self.embedding_dim:
            raise ValueError(
                "输入的`query`、`key`和`value`的维度（embedding_dim）和MHA模型预设维度不匹配，"
                f"模型预设维度为{self.embedding_dim}，而输入维度是{input_embedding_dim}"
            )

        queries: Tensor = self.q_weight(q_seq)
        keys: Tensor = self.k_weight(k_seq)
        values: Tensor = self.v_weight(v_seq)
        # 拆分多头
        head_dim = input_embedding_dim // self.head_count
        # 即最后一维拆分 -> embedding_dim = head_count * head_dim，并交换head_count和seq_dim维度
        queries = (
            queries.contiguous()
            .view(batch_size, src_seq_len, self.head_count, head_dim)
            .permute(0, 2, 1, 3)
        )
        keys = (
            keys.contiguous()
            .view(batch_size, tgt_seq_len, self.head_count, head_dim)
            .permute(0, 2, 1, 3)
        )
        values = (
            values.contiguous()
            .view(batch_size, tgt_seq_len, self.head_count, head_dim)
            .permute(0, 2, 1, 3)
        )
        # 计算注意力
        # mask = torch.tril(torch.ones(src_seq_len, src_seq_len, dtype=bool))
        attention_values, _ = _scaled_dot_product_attention(queries, keys, values, mask)
        # 合并多头
        attention_values = (
            attention_values.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, src_seq_len, input_embedding_dim)
        )
        return self.output_weight(attention_values)


class TokenEmbedding(nn.Embedding):
    """用来将Token转化为embedding，其实就是封装了一下nn.Embedding"""

    def __init__(self, vocab_size, embedding_dim, padding_token=0):
        super().__init__(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_token,
        )


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, seq_len, embedding_dim):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

        # 位置编码，不参与学习
        self.positional_encoding = torch.zeros(seq_len, embedding_dim)
        self.positional_encoding.requires_grad_(False)

        # pos和index一个列向量，一个行向量，
        # 在计算时经过python广播，就得到了一个(seq_len, embedding_dim)的矩阵
        pos = torch.arange(0, seq_len)
        pos = pos.float().unsqueeze(1)
        index = torch.arange(0, embedding_dim)
        index = pos.float().unsqueeze(0)
        _tmp = pos / torch.pow(10000, index / embedding_dim)
        self.positional_encoding[:, 0::2] = torch.sin(_tmp[:, 0::2])
        self.positional_encoding[:, 1::2] = torch.cos(_tmp[:, 1::2])

    def forward(self, input):
        """接受一个(seq_len, embedding_dim)的输入，把PE加给它"""
        input_seq_len = input.shape[-2]
        input_embedding_dim = input.shape[-1]
        if input_seq_len != self.seq_len or input_embedding_dim != self.embedding_dim:
            raise ValueError("[PositionEmbedding] 输入维度和模块维度不匹配")
        return self.positional_encoding


class TransformerEmbedding(nn.Module):
    """对 TokenEmbedding 和 PositionalEncoding 的整合"""

    def __init__(self, vocab_size, embedding_dim, seq_len, dropout_prob):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(seq_len, embedding_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input):
        output = self.token_embedding(input)
        output = output + self.positional_encoding(output)
        return self.dropout(output)


class LayerNorm(nn.Module):
    def __init__(self, channel_dim, epsilon=1e-5):
        super().__init__()
        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(channel_dim))
        self.beta = nn.Parameter(torch.zeros(channel_dim))
        # epsilon
        self.epsilon = epsilon

    def forward(self, input: Tensor):
        mean = input.mean(-1, keepdim=True)
        var = input.var(-1, unbiased=False, keepdim=True)
        output = (input - mean) / torch.sqrt(var + self.epsilon)
        output = self.gamma * output + self.beta
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_prob):
        super().__init__()
        self.full_connection_1 = nn.Linear(embedding_dim, hidden_dim)
        self.full_connection_2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input):
        output = self.full_connection_1(input)
        output = torch.relu(output)
        if self.dropout is not None:
            output = self.dropout(output)
        output = self.full_connection_2(output)
        return output


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_count, ffn_hidden_dim, dropout_prob):
        super().__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, head_count)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.norm1 = LayerNorm(embedding_dim)

        self.feedforward = PositionwiseFeedForward(
            embedding_dim, ffn_hidden_dim, dropout_prob
        )
        self.dropout2 = nn.Dropout(dropout_prob)
        self.norm2 = LayerNorm(embedding_dim)

    def forward(self, input: Tensor, src_mask: Tensor | None = None):
        _input = input
        input = input + self.self_attention(input, input, input, src_mask)
        input = self.dropout1(input)
        input = self.norm1(_input + input)  # 一个跳跃链接

        _input = input
        input = self.feedforward(input)
        input = self.dropout2(input)
        input = self.norm2(_input + input)
        return input


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_count, ffn_hidden_dim, dropout_prob):
        super().__init__()
        self.masked_attention = MultiHeadAttention(embedding_dim, head_count)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.norm1 = LayerNorm(embedding_dim)

        self.cross_attention = MultiHeadAttention(embedding_dim, head_count)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.norm2 = LayerNorm(embedding_dim)

        self.feedforward = PositionwiseFeedForward(
            embedding_dim, ffn_hidden_dim, dropout_prob
        )
        self.dropout3 = nn.Dropout(dropout_prob)
        self.norm3 = LayerNorm(embedding_dim)

    def forward(
        self,
        input: Tensor,
        encoder_output: Tensor,
        src_tgt_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ):
        _input = input
        seq_len = input.shape[-2]
        if tgt_mask is None:
            tgt_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=bool))
        input = input + self.masked_attention(input, input, input, tgt_mask)
        input = self.dropout1(input)
        input = self.norm1(_input + input)

        _input = input
        input = input + self.cross_attention(
            input, encoder_output, encoder_output, src_tgt_mask
        )
        input = self.dropout2(input)
        input = self.norm2(_input + input)

        _input = input
        input = self.feedforward(input)
        input = self.dropout3(input)
        input = self.norm3(_input + input)
        return input


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        src_seq_len,
        embedding_dim,
        head_count,
        ffn_hidden_dim,
        encoder_block_count,
        dropout_prob,
        device,
    ):
        super().__init__()
        self.embedding = TransformerEmbedding(
            src_vocab_size, embedding_dim, src_seq_len, dropout_prob
        )
        self.encoder_blocks = nn.ModuleList(
            EncoderBlock(embedding_dim, head_count, ffn_hidden_dim, dropout_prob)
            for _ in range(encoder_block_count)
        )

    def forward(self, input, src_mask):
        output = self.embedding(input)
        for block in self.encoder_blocks:
            output = block(output, src_mask)
        return output


class Decoder(nn.Module):
    def __init__(
        self,
        tgt_vocab_size,
        tgt_seq_len,
        embedding_dim,
        head_count,
        ffn_hidden_dim,
        decoder_block_count,
        dropout_prob,
        device,
    ):
        super().__init__()
        self.embedding = TransformerEmbedding(
            tgt_vocab_size, embedding_dim, tgt_seq_len, dropout_prob
        )
        self.decoder_blocks = nn.ModuleList(
            DecoderBlock(embedding_dim, head_count, ffn_hidden_dim, dropout_prob)
            for _ in range(decoder_block_count)
        )
        # 最后的线性层
        self.full_connection = nn.Linear(embedding_dim, tgt_vocab_size)

    def forward(self, input, encoder_output, src_tgt_mask, tgt_mask):
        output = self.embedding(input)
        for block in self.decoder_blocks:
            output = block(output, encoder_output, src_tgt_mask, tgt_mask)
        output = self.full_connection(output)
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        src_vacab_size,
        tgt_vacab_size,
        src_seq_len,
        tgt_seq_len,
        src_padding_idx,
        tgt_padding_idx,
        embedding_dim,
        head_count,
        ffn_hidden_dim,
        encoder_block_count,
        decoder_block_count,
        dropout_prob,
        device,
    ):
        self.encoder = Encoder(
            src_vacab_size,
            src_seq_len,
            embedding_dim,
            head_count,
            ffn_hidden_dim,
            encoder_block_count,
            dropout_prob,
            device,
        )
        self.decoder = Decoder(
            tgt_vacab_size,
            tgt_seq_len,
            embedding_dim,
            head_count,
            ffn_hidden_dim,
            decoder_block_count,
            dropout_prob,
            device,
        )

        self.src_padding_idx = src_padding_idx
        self.tgt_padding_idx = tgt_padding_idx
        self.device = device

    def make_padding_mask(
        self, query: Tensor, key: Tensor, query_padding_index, key_padding_index
    ):
        """生成 padding mask。在 embedding 之前使用。

        想想计算注意力时，Q 乘 K 转置的结果，它是一个 `(query_seq_len, key_seq_len)` 的矩阵，也就是说，
        横轴对应 Q，纵轴对应 K。那么，只要 Q 或者 K 其中有一者为 padding，该位置就需要 mask。

        Padding mask 的形状为 `(batch_size, 1, query_seq_len, key_seq_len)`，意义是：
        每一个句子的 padding 都不一样，所以要具体到 batch_size，而 head 是从 embedding 分出来的，
        每一个 head 对应的句子都是一样的，所以第二个维度设置为 1，利用广播机制复制到 head_count 维上。

        Args:
            query (Tensor): Q 矩阵，形状为 `(batch_size, query_seq_len)`
            key (Tensor): V 矩阵，形状为 `(batch_size, key_seq_len)`
            query_padding_index (int): Q 矩阵中，哪一个 token 代表 padding
            key_padding_index (int): V 矩阵中，哪一个 token 代表 padding
        """
        query_seq_len = query.shape[1]
        key_seq_len = key.shape[1]
        # 都扩展成 mask 的形状
        # query 扩展第 1 维和第 3 维，并将第 3 维重复到 key_seq_len 的数量
        query = query.ne(query_padding_index).unsqueeze(1).unsqueeze(3)
        query = query.repeat(1, 1, 1, key_seq_len)
        # query 扩展第 1 维和第 2 维，并将第 2 维重复到 query_seq_len 的数量
        key = key.ne(key_padding_index).unsqueeze(1).unsqueeze(2)
        key = key.repeat(1, 1, query_seq_len, 1)

        mask = query & key
        return mask

    def make_causal_mask(self, query: Tensor, key: Tensor):
        """生成 causal mask。在 embedding 之前使用。

        Causal mask 的形状为 `(query_seq_len, key_seq_len)`，和计算注意力时 Q 和 K 转置的乘积形状一样。
        横轴对应 Q，纵轴对应 K。

        Args:
            query (Tensor): 用作 query 的句子组成的矩阵，形状为 `(batch_size, query_seq_len)`
            key (Tensor): 用作 key 的句子组成的矩阵，形状为 `(batch_size, key_seq_len)`
        """
        query_seq_len = query.shape[1]
        key_seq_len = key.shape[1]
        mask = torch.tril(torch.ones(query_seq_len, key_seq_len, dtype=bool)).to(
            self.device
        )
        return mask

    def forward(self, src, tgt):
        # ? Encoder 处的自注意力模块，用到了 padding mask
        src_mask = self.make_padding_mask(
            src, src, self.src_padding_idx, self.src_padding_idx
        )

        # ? Decoder 处的自注意力模块，既有 padding mask，又有 causal mask
        _tgt_padding_mask = self.make_padding_mask(
            tgt, tgt, self.tgt_padding_idx, self.tgt_padding_idx
        )
        _tgt_causal_mask = self.make_causal_mask(tgt, tgt)
        # padding_mask 和 causal_mask 逐元素相乘
        tgt_mask = _tgt_padding_mask * _tgt_causal_mask

        # ? Encoder 和 Decoder 的交叉注意力模块，用到了 padding mask
        src_tgt_mask = self.make_padding_mask(
            src, tgt, self.src_padding_idx, self.tgt_padding_idx
        )

        encoder_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, encoder_output, src_tgt_mask, tgt_mask)
        return output
