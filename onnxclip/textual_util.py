import torch
from torch import nn
import torch.nn.functional as F
from open_clip.transformer import text_global_pool


class TextualWrapper(nn.Module):
    def __init__(self, clip_model: nn.Module) -> None:
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.text_pool_type = clip_model.text_pool_type
        self.attn_mask = clip_model.attn_mask


    def forward(self, input_ids: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(input_ids).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, input_ids, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x