from __future__ import annotations

import torch
from torch import nn

from tavr_vlm.models.decoder import TransformerReportDecoder
from tavr_vlm.models.encoders import CT3DEncoder, EchoTemporalEncoder, FusionEncoder, TabularEncoder
from tavr_vlm.models.rcga import RCGAModule


class TAVRVLM(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        mcfg = cfg["model"]
        vocab_size = int(mcfg["vocab_size"])
        hidden = int(mcfg.get("hidden_dim", 512))
        ct_cfg = mcfg["ct"]
        echo_cfg = mcfg["echo"]
        tab_cfg = mcfg["tabular"]
        fusion_cfg = mcfg["fusion"]
        rcga_cfg = mcfg["rcga"]
        dec_cfg = mcfg["decoder"]

        self.ct_encoder = CT3DEncoder(
            in_channels=ct_cfg.get("in_channels", 1),
            patch_size=tuple(ct_cfg.get("patch_size", [8, 16, 16])),
            embed_dim=ct_cfg.get("embed_dim", hidden),
            depth=ct_cfg.get("depth", 4),
            num_heads=ct_cfg.get("num_heads", 8),
            dropout=ct_cfg.get("dropout", 0.1),
        )
        self.echo_encoder = EchoTemporalEncoder(
            in_channels=echo_cfg.get("in_channels", 1),
            embed_dim=echo_cfg.get("embed_dim", 256),
            temporal_layers=echo_cfg.get("temporal_layers", 2),
            dropout=echo_cfg.get("dropout", 0.1),
        )
        self.tabular_encoder = TabularEncoder(
            input_dim=mcfg.get("tabular_dim", 32),
            hidden_dims=tab_cfg.get("hidden_dims", [128, 256]),
            output_dim=hidden,
            dropout=tab_cfg.get("dropout", 0.1),
        )
        self.fusion = FusionEncoder(
            ct_dim=ct_cfg.get("embed_dim", hidden),
            echo_dim=echo_cfg.get("embed_dim", 256),
            tab_dim=hidden,
            hidden_dim=fusion_cfg.get("hidden_dim", hidden),
            dropout=fusion_cfg.get("dropout", 0.1),
        )
        self.rcga = RCGAModule(
            hidden_dim=fusion_cfg.get("hidden_dim", hidden),
            visual_dim=ct_cfg.get("embed_dim", hidden),
            risk_classes=mcfg.get("risk_classes", 3),
            prototype_dim=rcga_cfg.get("prototype_dim", hidden),
            topk_ratio=rcga_cfg.get("topk_ratio", 0.2),
            disable_risk_prototypes=rcga_cfg.get("disable_risk_prototypes", False),
            disable_purification=rcga_cfg.get("disable_purification", False),
            stop_gradient_global=rcga_cfg.get("stop_gradient_global", True),
        )
        self.decoder = TransformerReportDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden,
            num_layers=dec_cfg.get("num_layers", 4),
            num_heads=dec_cfg.get("num_heads", 8),
            ff_dim=dec_cfg.get("ff_dim", 2048),
            dropout=dec_cfg.get("dropout", 0.1),
            pad_token_id=mcfg.get("pad_token_id", 0),
        )
        self.recommendation_head = nn.Linear(hidden, int(mcfg.get("recommendation_dim", 8)))
        self.pad_token_id = int(mcfg.get("pad_token_id", 0))
        self.bos_token_id = int(mcfg.get("bos_token_id", 1))
        self.eos_token_id = int(mcfg.get("eos_token_id", 2))
        self.max_report_len = int(mcfg.get("max_report_len", 160))

    def encode(self, ct: torch.Tensor, echo: torch.Tensor, tabular: torch.Tensor) -> dict[str, torch.Tensor]:
        v_visual, ct_pool = self.ct_encoder(ct)
        echo_pool = self.echo_encoder(echo)
        tab_pool = self.tabular_encoder(tabular)
        h = self.fusion(ct_pool, echo_pool, tab_pool)
        rcga = self.rcga(h, v_visual)
        rcga["joint_embedding"] = h
        rcga["v_visual"] = v_visual
        rcga["recommendation_logits"] = self.recommendation_head(h)
        return rcga

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        enc = self.encode(batch["ct"], batch["echo"], batch["tabular"])
        report = batch["report_ids"]
        decoder_in = report[:, :-1]
        logits, hidden = self.decoder(decoder_in, enc["v_act"])
        m_t, m_tilde = self.rcga.token_grounding(hidden, enc["v_visual"], enc["b_global"])
        enc.update({
            "lm_logits": logits,
            "decoder_hidden": hidden,
            "m_t": m_t,
            "m_tilde": m_tilde,
        })
        return enc

    @torch.no_grad()
    def generate(self, batch: dict[str, torch.Tensor], max_len: int | None = None) -> dict[str, torch.Tensor]:
        max_len = max_len or self.max_report_len
        enc = self.encode(batch["ct"], batch["echo"], batch["tabular"])
        b = batch["ct"].shape[0]
        ids = torch.full((b, 1), self.bos_token_id, dtype=torch.long, device=batch["ct"].device)
        hidden = None
        for _ in range(max_len - 1):
            logits, hidden = self.decoder(ids, enc["v_act"])
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if torch.all(nxt.squeeze(1).eq(self.eos_token_id)):
                break
        if hidden is None:
            logits, hidden = self.decoder(ids, enc["v_act"])
        m_t, m_tilde = self.rcga.token_grounding(hidden, enc["v_visual"], enc["b_global"])
        enc.update({"generated_ids": ids, "m_t": m_t, "m_tilde": m_tilde})
        return enc
