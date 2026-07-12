**🌐 Language**：[中文](./README.md) | [English](#)

# Jasper-Token-Compression-600M

> This is a mirror/fork of the original [infgrad/Jasper-Token-Compression-600M](https://huggingface.co/infgrad/Jasper-Token-Compression-600M) repository. All credit goes to the original authors: Dun Zhang, Ziyang Zeng, Yudong Zhou, Shuyang Lu.

---

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
