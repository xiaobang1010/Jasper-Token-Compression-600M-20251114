**🌐 Language**：[中文](./README.md) | [English](#)

# Jasper-Token-Compression-600M

> Mirror of [infgrad/Jasper-Token-Compression-600M](https://huggingface.co/infgrad/Jasper-Token-Compression-600M). Original authors: Dun Zhang, Ziyang Zeng, Yudong Zhou, Shuyang Lu.

A bilingual (EN/ZH) text embedding model with dynamic token compression (up to 10x). Combines vector distillation and contrastive learning. Compatible with sentence-transformers.

- 📄 Paper: [arXiv 2511.14405](https://arxiv.org/abs/2511.14405)
- 🔧 Training code: [Jasper-Token-Compression-Training](https://github.com/DunZhang/Jasper-Token-Compression-Training)

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("infgrad/Jasper-Token-Compression-600M", trust_remote_code=True)

query_embeddings = model.encode(["What is photosynthesis?"], normalize_embeddings=True)
document_embeddings = model.encode(["Photosynthesis is the process by which green plants..."], normalize_embeddings=True)

similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)
```

## License

MIT License · All rights belong to the original authors
