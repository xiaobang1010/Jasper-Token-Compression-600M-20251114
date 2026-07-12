**🌐 语言 / Language**：[中文](#) | [English](./README_EN.md)

# Jasper-Token-Compression-600M

> 本仓库是 [infgrad/Jasper-Token-Compression-600M](https://huggingface.co/infgrad/Jasper-Token-Compression-600M) 的镜像。原作者：Dun Zhang, Ziyang Zeng, Yudong Zhou, Shuyang Lu。

支持中英双语的文本向量模型，采用动态 Token 压缩技术，最高可将文本压缩至原来的 1/10，同时保持优异性能。结合向量蒸馏与对比学习，兼容 sentence-transformers。

- 📄 论文：[arXiv 2511.14405](https://arxiv.org/abs/2511.14405)
- 🔧 训练代码：[Jasper-Token-Compression-Training](https://github.com/DunZhang/Jasper-Token-Compression-Training)

## 快速使用

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("infgrad/Jasper-Token-Compression-600M", trust_remote_code=True)

query_embeddings = model.encode(["什么是光合作用？"], normalize_embeddings=True)
document_embeddings = model.encode(["光合作用是绿色植物利用光能..."], normalize_embeddings=True)

similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)
```

## 许可

MIT License · 版权归原作者所有
