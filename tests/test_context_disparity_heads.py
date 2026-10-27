import sys,copy
sys.path.insert(0,'.')
import torch
from src.model.encoder.costvolume.depth_predictor_multiview import DepthPredictorMultiView

torch.set_num_threads(2)
model=DepthPredictorMultiView(alternate_context_heads=True,feature_channels=8,depth_unet_feat_dim=32,gaussian_raw_channels=10,num_views=2,num_depth_candidates=4,depth_unet_channel_mult=(1,),depth_unet_attn_res=())
heads=model.to_disparity
assert len(heads)==2
assert all(isinstance(m,(torch.nn.Sequential,torch.nn.Conv2d,torch.nn.GELU)) for head in heads for m in head.modules())
legacy={k:v.clone() for k,v in model.state_dict().items() if not k.startswith(('to_disparity.', 'to_gaussians.'))}
legacy.update({'to_disparity.'+k:v.clone() for k,v in heads[0].state_dict().items()})
legacy.update({'to_gaussians.'+k:v.clone() for k,v in model.to_gaussians[0].state_dict().items()})
model.load_state_dict(legacy,strict=True)
gx=torch.randn(6,43,7,9)
gheads=model.to_gaussians
for p,q in zip(gheads[0].parameters(),gheads[1].parameters()):
 assert torch.equal(p,q) and p.data_ptr()!=q.data_ptr()
torch.testing.assert_close(torch.cat([h(gx[i*3:(i+1)*3]) for i,h in enumerate(gheads)]), gheads[0](gx))
assert all(isinstance(m,(torch.nn.Sequential,torch.nn.Conv2d,torch.nn.GELU)) for h in gheads for m in h.modules())
for p,q in zip(heads[0].parameters(),heads[1].parameters()):
 assert torch.equal(p,q) and p.data_ptr()!=q.data_ptr()
x=torch.randn(6,32,7,9,requires_grad=True)
old=heads[0](x)
new=torch.cat([head(x[i*3:(i+1)*3]) for i,head in enumerate(heads)])
torch.testing.assert_close(new,old,rtol=1e-5,atol=1e-6)
new[:3].sum().backward()
assert all(p.grad is None or p.grad.count_nonzero()==0 for p in heads[1].parameters())
assert x.grad[3:].count_nonzero()==0
clone=copy.deepcopy(model);clone.load_state_dict(model.state_dict(),strict=True)
# Exercise the production forward slicing with B=2 (view-major ordering).
calls=[]
gcalls=[]
ghandles=[h.register_forward_pre_hook(lambda m,a,i=i: gcalls.append((i,a[0].shape[0]))) for i,h in enumerate(gheads)]
handles=[h.register_forward_pre_hook(lambda m,a,i=i: calls.append((i,a[0].shape[0]))) for i,h in enumerate(heads)]
b,v=2,2
extra={'images':torch.rand(v*b,3,16,16)}
pose=torch.eye(4).repeat(b,v,1,1);pose[:,1,0,3]=.1
out=model(torch.rand(b,v,8,4,4),torch.eye(3).repeat(b,v,1,1),pose,torch.ones(b,v),torch.full((b,v),80.),extra_info=extra,cnn_features=torch.rand(b,v,8,4,4))
assert calls==[(0,2),(1,2)],calls
assert gcalls==[(0,2),(1,2)],gcalls
assert out[0].shape==(2,2,256,1,1)
print('PASS: legacy migration, independent storage, equivalent outputs, isolated head gradients, new checkpoint reload, production forward B=2, Conv/GELU only')

# Disabled alternation constructs true shared Sequential heads, with legacy keys.
shared=DepthPredictorMultiView(feature_channels=8,depth_unet_feat_dim=32,gaussian_raw_channels=10,num_views=2,num_depth_candidates=4,depth_unet_channel_mult=(1,),depth_unet_attn_res=())
assert isinstance(shared.to_disparity, torch.nn.Sequential)
assert isinstance(shared.to_gaussians, torch.nn.Sequential)
shared.load_state_dict(legacy, strict=True)
assert sum(p.numel() for p in shared.to_disparity.parameters())*2 == sum(p.numel() for p in heads.parameters())
for name, inp in [('to_disparity', x.detach()), ('to_gaussians', gx)]:
    head=getattr(shared,name)
    torch.testing.assert_close(shared._apply_context_head(head,inp,3,2),head(inp))
shared_calls=[]
hook=shared.to_disparity.register_forward_pre_hook(lambda m,a: shared_calls.append(a[0].shape[0]))
shared(torch.rand(b,v,8,4,4),torch.eye(3).repeat(b,v,1,1),pose,torch.ones(b,v),torch.full((b,v),80.),extra_info=extra,cnn_features=torch.rand(b,v,8,4,4))
assert shared_calls==[4], shared_calls
hook.remove()
try:
    shared.load_state_dict(model.state_dict(),strict=True)
except RuntimeError as error:
    assert 'split-head checkpoint' in str(error)
else:
    raise AssertionError('Split weights must not silently initialize shared heads')
print('PASS: disabled alternation uses shared heads, legacy loading, full-view forward, split checkpoint rejection')
