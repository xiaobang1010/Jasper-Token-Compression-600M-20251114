---
license: mit
tags:
- sentence-transformers
- sentence-similarity
- mteb
- retriever
- text-embeddings-inference
language:
- en
- zh
base_model:
- Qwen/Qwen3-Embedding-0.6B
library_name: adapter-transformers
---

**🌐 语言切换 / Language**：[中文（默认）](#) | [English](#english-version)

# Jasper-Token-Compression-600M

> **注意**：本仓库是原项目 [infgrad/Jasper-Token-Compression-600M](https://huggingface.co/infgrad/Jasper-Token-Compression-600M) 的镜像/Fork。所有荣誉归原作者：Dun Zhang, Ziyang Zeng, Yudong Zhou, Shuyang Lu。

---

## 📖 简介

受 Deepseek-OCR 启发，这是 Jasper 和 Stella 系列中首个使用**动态文本 Token 压缩**技术的向量模型。通过向量蒸馏与对比学习的结合，我们的模型可以将文本压缩至原来的 1/10，同时仍能取得优异的性能表现！

原始训练代码：https://github.com/DunZhang/Jasper-Token-Compression-Training

论文报告：https://arxiv.org/abs/2511.14405

微信：zhdunt

X（Twitter）：https://x.com/dunn_zhang

## ✨ 特性

- ⭐⭐⭐ 支持中英双语
- ⭐⭐⭐⭐⭐⭐ 动态 Token 压缩 — 经测试，即使将文本压缩到原始长度的 0.33 倍，仍能取得优异结果
- ⭐⭐⭐ 结合向量蒸馏与对比学习，进一步提升检索任务性能
- ⭐⭐ 1200 万无监督数据蒸馏
- ⭐⭐ 0.6B 参数量

## 🔧 技术细节

### 动态文本 Token 压缩

我的实现非常简单：文本经过 `word_embedding` 层后，立即进入一个 `Qwen3MLP`（约 3 个全连接层），然后计算压缩后的长度，最后使用 `adaptive_avg_pool1d` 将 Token 压缩到目标长度。

压缩长度计算逻辑如下：

```python
real_length = 1000  # 文本实际 Token 数
length_threshold = 80  # 仅当超过此阈值时才压缩
compression_ratio = 0.333
if real_length <= length_threshold:
    # 不进行压缩
    pass
else:
    target_length = int(length_threshold + (real_length - length_threshold) * compression_ratio)
```

具体实现细节请参阅本目录下的 `modeling_qwen3_jasper.py` 文件。

### 向量蒸馏 + 对比学习

首先，我们在对比学习训练集中为每个文本计算教师向量，然后在训练过程中使用以下三种损失函数：

1. **余弦损失（Cosine Loss）**：标准向量蒸馏损失
2. **InfoNCE（硬损失）**：标准对比学习损失函数
3. **KL 散度（软损失）**：学生模型分数矩阵与教师模型分数矩阵之间的 KL 散度。分数矩阵是查询与所有文档（正例文档、难负例文档、其他批内文档）之间的分数

#### 评估

我的 Prompt 策略和具体内容与 QZhou 模型一致，详情请参考其评估脚本：https://github.com/Kingsoft-LLM/QZhou-Embedding

### 使用方法

```py
import torch
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    model_name_or_path = "infgrad/Jasper-Token-Compression-600M"
    model = SentenceTransformer(
        model_name_or_path,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",  # 支持 flash_attention_2; sdpa; eager
            "trust_remote_code": True
        },
        trust_remote_code=True,
        tokenizer_kwargs={"padding_side": "left"},
        device="cpu",
    )

    queries = [
        "What is photosynthesis?",
        "Who invented the telephone?",
    ]
    documents = [
        "Photosynthesis is the process by which green plants use sunlight, carbon dioxide, and water to produce glucose and oxygen",
        "Alexander Graham Bell is credited with inventing the first practical telephone in 1876, receiving US patent number 174,465 for his device."
    ]
    # compression_ratio 参数越小，速度越快，但质量会相应下降。
    # 根据我们训练时的参数设置和测试结果，建议范围为 0.3-0.8。
    query_embeddings = model.encode(queries, prompt_name="query", normalize_embeddings=True, compression_ratio=0.3333)
    document_embeddings = model.encode(documents, normalize_embeddings=True, compression_ratio=0.3333)

    similarity = model.similarity(query_embeddings, document_embeddings)
    print(similarity)

```

## ⚠️ 局限性与 TODO

### 检索性能

我发现蒸馏模型很难接近教师模型的检索性能，因此我特别使用了对比学习 + 蒸馏学习来增强学生模型。然而，虽然增强后的模型在检索测试集上有所提升，但与主流模型相比仍有显著差距。
**因此，我认为如何提升蒸馏模型的检索性能是一个非常有价值且必要的研究方向。**

### 更合理的文本 Token 压缩模块

目前关于文本 Token 压缩的研究还比较有限，我只尝试了最简单的方案。我相信一定存在更合理的文本压缩模块。

### 文本长度

我仅对最长 1024 个 Token 的文本进行了蒸馏，因此当文本长度超过 1024 时，性能可能会下降。

### 引用

如果你觉得我们的工作值得引用，请使用以下引用格式。

```

@misc{zhang2025jasperstelladistillationsota,
      title={Jasper and Stella: distillation of SOTA embedding models}, 
      author={Dun Zhang and Jiacheng Li and Ziyang Zeng and Fulong Wang},
      year={2025},
      eprint={2412.19048},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2412.19048}, 
}

```

```
@misc{zhang2025jaspertokencompression600mtechnicalreport,
      title={Jasper-Token-Compression-600M Technical Report}, 
      author={Dun Zhang and Ziyang Zeng and Yudong Zhou and Shuyang Lu},
      year={2025},
      eprint={2511.14405},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2511.14405}, 
}
```

---

<a id="english-version"></a>

> This is a mirror/fork of the original [infgrad/Jasper-Token-Compression-600M](https://huggingface.co/infgrad/Jasper-Token-Compression-600M) repository. All credit goes to the original authors: Dun Zhang, Ziyang Zeng, Yudong Zhou, Shuyang Lu.

## 📖 Introduction

Inspired by Deepseek-OCR, this is the first vector model in the Jasper and Stella series to use **dynamic text token
compression**. Through the combination of vector distillation and contrastive learning, our model can compress text by
10x
while still achieving excellent performance!

Original training codes: https://github.com/DunZhang/Jasper-Token-Compression-Training

Report: https://arxiv.org/abs/2511.14405

Wechat: zhdunt

X: https://x.com/dunn_zhang

## ✨ Features

- ⭐⭐⭐ Supports bilingual (Chinese and English)
- ⭐⭐⭐⭐⭐⭐ Dynamic token compression - tested to achieve excellent results even when compressing text to 0.33x of original
  length
- ⭐⭐⭐ Combines vector distillation with contrastive learning to further improve performance on retrieval tasks
- ⭐⭐ 12 million unsupervised data distillation
- ⭐⭐ 0.6B parameter size

## 🔧 Technical Details

### Dynamic Text Token Compression

My implementation is very simple: After text passes through the `word_embedding` layer, it immediately goes through a
`Qwen3MLP` (approximately 3 fully connected layers), then I calculate the compressed length, and finally use
`adaptive_avg_pool1d` to compress tokens to that length.

The compression length calculation logic is as follows:

```python
real_length = 1000  # Actual token count of the text
length_threshold = 80  # Compress only if exceeding this threshold
compression_ratio = 0.333
if real_length <= length_threshold:
    # No compression
    pass
else:
    target_length = int(length_threshold + (real_length - length_threshold) * compression_ratio)
```

For implementation details, please refer to the `modeling_qwen3_jasper.py` file in this directory.

### Vector Distillation + Contrastive Learning

First, we compute teacher vectors for texts in the contrastive learning training set, then use the following three
losses during training:

1. Cosine loss: Standard vector distillation loss
2. InfoNCE (hard loss): Standard contrastive learning loss function
3. KL divergence (soft loss): KL divergence between student score matrix and teacher score matrix. The score matrix is
   the scores between query and all documents(i.e. positive doc, hard negative docs, other in-batch docs).

#### Evaluation

My prompt strategy and specific content are consistent with the QZhou model. Please refer to their evaluation
script: https://github.com/Kingsoft-LLM/QZhou-Embedding

### Usage

```py
import torch
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    model_name_or_path = "infgrad/Jasper-Token-Compression-600M"
    model = SentenceTransformer(
        model_name_or_path,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",  # We support flash_attention_2; sdpa; eager
            "trust_remote_code": True
        },
        trust_remote_code=True,
        tokenizer_kwargs={"padding_side": "left"},
        device="cpu",
    )

    queries = [
        "What is photosynthesis?",
        "Who invented the telephone?",
    ]
    documents = [
        "Photosynthesis is the process by which green plants use sunlight, carbon dioxide, and water to produce glucose and oxygen",
        "Alexander Graham Bell is credited with inventing the first practical telephone in 1876, receiving US patent number 174,465 for his device."
    ]
    # The smaller the compression_ratio parameter, the faster the speed, but the quality will correspondingly decrease.
    # Based on our parameter settings during training and test results, we recommend a range between 0.3-0.8.
    query_embeddings = model.encode(queries, prompt_name="query", normalize_embeddings=True, compression_ratio=0.3333)
    document_embeddings = model.encode(documents, normalize_embeddings=True, compression_ratio=0.3333)

    similarity = model.similarity(query_embeddings, document_embeddings)
    print(similarity)

```

## ⚠️ Limitations and TODO

### Retrieval performance

I found that distilled models struggle to approach the retrieval performance of teacher models, which is why I
specifically used contrastive learning + distillation learning to enhance the student model. However, I found that while
the enhanced model showed improvement on retrieval test sets, there is still a significant gap compared to mainstream
models.
**Therefore, I believe that how to improve the retrieval performance of distilled models is a very necessary and
valuable
research direction.**

### More reasonable text token compression modules

There is limited research on text token compression currently, and I have only tried the simplest approach. I believe
more reasonable text compression modules can definitely be found.

### Text length

I only distilled texts up to 1024 tokens in length, so there should be performance degradation when text length exceeds
1024.

### Citation

If you find our work worth citing, please use the following citation.

```

@misc{zhang2025jasperstelladistillationsota,
      title={Jasper and Stella: distillation of SOTA embedding models}, 
      author={Dun Zhang and Jiacheng Li and Ziyang Zeng and Fulong Wang},
      year={2025},
      eprint={2412.19048},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2412.19048}, 
}

```

```
@misc{zhang2025jaspertokencompression600mtechnicalreport,
      title={Jasper-Token-Compression-600M Technical Report}, 
      author={Dun Zhang and Ziyang Zeng and Yudong Zhou and Shuyang Lu},
      year={2025},
      eprint={2511.14405},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2511.14405}, 
}
```
