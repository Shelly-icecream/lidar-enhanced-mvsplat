"""CPU regression: config, freezing, switch boundaries, and Adam state isolation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from hydra import compose, initialize_config_dir
from src.config import load_typed_root_config
from src.global_cfg import set_cfg
from src.model.encoder import get_encoder


def test_alternating_context_heads():
    torch.set_num_threads(2)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'config'), version_base=None):
        config = compose(config_name='main', overrides=['+experiment=stage5'])
    set_cfg(config)
    cfg = load_typed_root_config(config)
    encoder, _ = get_encoder(cfg.model.encoder)
    heads = torch.nn.ModuleList([torch.nn.ModuleList([d, g]) for d, g in zip(
        encoder.depth_predictor.to_disparity, encoder.depth_predictor.to_gaussians)])
    assert cfg.model.encoder.alternate_context_heads
    assert cfg.model.encoder.alternate_context_head_steps == 500
    assert all(not p.requires_grad for n, p in encoder.named_parameters()
               if not n.startswith(('depth_predictor.to_disparity.', 'depth_predictor.to_gaussians.')))
    encoder.train()
    assert encoder.training and not encoder.backbone.training
    assert not encoder.depth_predictor.refine_unet.training
    assert encoder.depth_predictor.to_gaussians.training
    assert all(h.training for h in heads)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=.1)
    x = torch.randn(1, cfg.model.encoder.depth_unet_feat_dim, 4, 4)
    gx = torch.randn(1, cfg.model.encoder.depth_unet_feat_dim + 3 + cfg.model.encoder.d_feature, 4, 4)
    frozen = {n: p.detach().clone() for n, p in encoder.named_parameters()
              if not n.startswith(('depth_predictor.to_disparity.', 'depth_predictor.to_gaussians.'))}
    # Both heads acquire Adam momentum, then each must remain exactly unchanged
    # during its inactive phase. Boundary numbers are optimizer global steps.
    for step, expected in [(0, 0), (499, 0), (500, 1), (999, 1), (1000, 0), (1500, 1)]:
        optimizer.zero_grad(set_to_none=False)
        assert encoder.select_context_head(step) == expected
        before = [[p.detach().clone() for p in h.parameters()] for h in heads]
        sum(head(x if j == 0 else gx).square().mean()
            for pair in heads for j, head in enumerate(pair)).backward()
        assert all(p.grad is None for p in heads[1-expected].parameters())
        optimizer.step()
        assert all(torch.equal(a, p) for a, p in zip(before[1-expected], heads[1-expected].parameters()))
        assert any(not torch.equal(a, p) for a, p in zip(before[expected], heads[expected].parameters()))
    assert all(torch.equal(frozen[n], p) for n, p in encoder.named_parameters() if n in frozen)
    encoder.eval()
    assert not any(m.training for m in encoder.modules())
    encoder.train()
    assert not encoder.depth_predictor.refine_unet.training
    # Alternation alone must not implicitly freeze the shared network.
    cfg.model.encoder.frozen_params = []
    unfrozen, _ = get_encoder(cfg.model.encoder)
    unfrozen.select_context_head(0)
    assert all(p.requires_grad for p in unfrozen.backbone.parameters())
    assert all(p.requires_grad for p in unfrozen.depth_predictor.refine_unet.parameters())
    assert unfrozen.backbone.training and unfrozen.depth_predictor.refine_unet.training
    print('PASS: stage5 config, frozen upstream, alternating boundaries, inactive Adam momentum/decay isolation')


if __name__ == '__main__':
    test_alternating_context_heads()
