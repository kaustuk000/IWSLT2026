import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, AutoProcessor


class MMSEncoder(nn.Module):
    def __init__(self, model_name="facebook/mms-1b-all"):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.encoder   = Wav2Vec2Model.from_pretrained(model_name, torch_dtype=torch.float16)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, input_values, attention_mask=None):
        with torch.no_grad():
            out = self.encoder(input_values.to(torch.float16), attention_mask=attention_mask)
        return out.last_hidden_state


class QFormer(nn.Module):
    def __init__(self, num_queries=80, encoder_dim=1280, qformer_dim=768,
                 num_heads=8, num_layers=6, llm_dim=4096, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, qformer_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.encoder_proj = nn.Sequential(
            nn.Linear(encoder_dim, qformer_dim),
            nn.LayerNorm(qformer_dim),
        )
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=qformer_dim, nhead=num_heads,
                dim_feedforward=qformer_dim * 4,
                dropout=dropout, batch_first=True, norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(qformer_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )

    def forward(self, encoder_hidden_states):
        B = encoder_hidden_states.size(0)
        memory  = self.encoder_proj(encoder_hidden_states)
        queries = self.queries.expand(B, -1, -1)
        out     = self.transformer(queries, memory)
        return self.output_proj(out)
