import torch
from sentence_transformers.models import Transformer as BaseTransformer


class JasperTransformer(BaseTransformer):
    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        vectors = self.auto_model(**features, **kwargs)
        features.update({"sentence_embedding": vectors})
        return features
