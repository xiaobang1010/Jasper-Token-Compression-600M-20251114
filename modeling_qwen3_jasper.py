import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3PreTrainedModel, Qwen3Config, Qwen3Model
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP


class TokenCompressor(nn.Module):
    """
    自适应Token压缩模块
    对于长度超过阈值的序列，使用adaptive_avg_pool1d进行压缩
    压缩后长度 = 阈值 + 超出部分 * compression_ratio
    """

    def __init__(self, length_threshold: int = 512, compression_ratio: float = 0.3):
        super().__init__()
        self.length_threshold = length_threshold
        self.compression_ratio = compression_ratio

    def forward(
            self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对token embeddings进行自适应压缩
        Args:
            token_embeddings: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        Returns:
            compressed_embeddings: 压缩后的embeddings
            compressed_mask: 压缩后的attention mask
        """
        padding_side = 'right' if (attention_mask[:, -1] == 0).any() else 'left'

        compressed_embeddings_list = []
        compressed_masks_list = []
        for text_idx in range(token_embeddings.shape[0]):
            # 获取当前样本的有效长度
            real_length = int(attention_mask[text_idx].sum().item())
            if real_length <= self.length_threshold:
                # 根据padding方向提取有效的token embeddings
                if padding_side == 'left':
                    # 左填充：有效tokens在右边
                    valid_embeddings = token_embeddings[text_idx:text_idx + 1, -real_length:, :]
                else:
                    # 右填充：有效tokens在左边
                    valid_embeddings = token_embeddings[text_idx:text_idx + 1, :real_length, :]
                compressed_embeddings_list.append(valid_embeddings)
                compressed_masks_list.append([1] * real_length)
            else:
                target_length = int(
                    self.length_threshold + (real_length - self.length_threshold) * self.compression_ratio
                )
                # 根据padding方向提取有效的token embeddings
                if padding_side == 'left':
                    # 左填充：有效tokens在右边
                    valid_embeddings = token_embeddings[text_idx:text_idx + 1, -real_length:, :]
                else:
                    # 右填充：有效tokens在左边
                    valid_embeddings = token_embeddings[text_idx:text_idx + 1, :real_length, :]

                # 使用adaptive_avg_pool1d进行压缩
                compressed_embeddings_list.append(
                    F.adaptive_avg_pool1d(
                        valid_embeddings.transpose(1, 2), target_length
                    ).transpose(1, 2)
                )
                # print("valid_embeddings.shape,target_length,compressed_embeddings_list[-1].shape",valid_embeddings.shape,target_length,compressed_embeddings_list[-1].shape)
                compressed_masks_list.append([1] * target_length)

        # 重新组合为token_embeddings和attention_mask
        new_seq_len = max((len(_mask) for _mask in compressed_masks_list))
        new_attention_mask = torch.tensor(
            [
                _mask + [0] * (new_seq_len - len(_mask))
                if padding_side == "right"
                else
                [0] * (new_seq_len - len(_mask)) + _mask
                for _mask in compressed_masks_list
            ],
            dtype=torch.long,
            device=token_embeddings.device
        )

        # 生成新的token_embeddings
        batch_size = token_embeddings.shape[0]
        hidden_size = token_embeddings.shape[2]
        new_token_embeddings = torch.zeros(
            batch_size, new_seq_len, hidden_size,
            dtype=token_embeddings.dtype,
            device=token_embeddings.device
        )

        for idx, compressed_emb in enumerate(compressed_embeddings_list):
            seq_len = compressed_emb.shape[1]
            if padding_side == "right":
                new_token_embeddings[idx, :seq_len, :] = compressed_emb.squeeze(0)
            else:
                # print("new_token_embeddings.shape,compressed_emb.shape",new_token_embeddings.shape,compressed_emb.shape)
                new_token_embeddings[idx, -seq_len:, :] = compressed_emb.squeeze(0)

        return new_token_embeddings, new_attention_mask



class JasperV2Encoder(Qwen3PreTrainedModel):

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.jasper_mlp = Qwen3MLP(config=config)
        self.linear_1 = nn.Linear(in_features=config.hidden_size, out_features=2048, bias=True)
        self.token_compressor = TokenCompressor(length_threshold=80, compression_ratio=0.5)
        self.post_init()

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            *args,
            **kwargs
    ) -> torch.Tensor:
        # token_embeddings.shape batch_size*seq_len*hidden_size
        token_embeddings = self.model.embed_tokens(input_ids)
        token_embeddings = self.jasper_mlp(token_embeddings)

        self.token_compressor.compression_ratio = kwargs.get(
            "compression_ratio",
            self.token_compressor.compression_ratio
        )
        compressed_token_embeddings, attention_mask = self.token_compressor(token_embeddings, attention_mask)
        compressed_token_embeddings = self.model(
            inputs_embeds=compressed_token_embeddings, attention_mask=attention_mask
        )["last_hidden_state"]

        # 生成句向量
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(compressed_token_embeddings.size()).to(
                compressed_token_embeddings.dtype)
        )
        sum_embeddings = torch.sum(compressed_token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        vector = sum_embeddings / sum_mask
        return self.linear_1(vector)
