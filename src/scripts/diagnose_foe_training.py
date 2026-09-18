"""Isolated original-MVSplat RGB-only controlled pilot; no LiDAR loading.

Run from repository root. Original source snapshot must exist in output/original.
"""
import argparse
import json
import sys
from pathlib import Path
from dataclasses import fields

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/foe_training"
ORIGINAL = OUT / "original"
sys.path.insert(0, str(OUT / "original"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from omegaconf import OmegaConf
from dacite import from_dict
from lpips import LPIPS
from src.global_cfg import set_cfg
from src.dataset.shims.crop_shim import rescale_and_crop
from src.model.encoder.encoder_costvolume import EncoderCostVolume, EncoderCostVolumeCfg
from src.model.decoder.cuda_splatting import render_cuda
from src.model.types import Gaussians
from src.model.encoder.common.gaussians import build_covariance


class OutputClamp:
    """Cache step-zero outputs per input; replace values, not module weights."""
    def __init__(self, model, mode):
        self.mode=mode; self.cache={}; self.recording=True; self.key=None
        self.max_error=0.; self.calls=0
        self.pre=model.register_forward_pre_hook(self.before)
        self.depth_hook=model.depth_predictor.register_forward_hook(self.depth)
        self.original_adapter=model.gaussian_adapter.forward
        model.gaussian_adapter.forward=self.adapter

    def before(self,module,inputs): self.key=inputs[0]["_sample_key"]

    def depth(self,module,inputs,output):
        if self.recording:
            self.cache.setdefault(self.key,{})["depth"]=output[0].detach().clone()
        elif self.mode=="depth":
            output=(self.cache[self.key]["depth"],*output[1:])
            assert not output[0].requires_grad
            self.max_error=max(self.max_error,float((output[0]-self.cache[self.key]["depth"]).abs().max()))
            self.calls+=1
        return output

    def adapter(self,*args,**kwargs):
        output=self.original_adapter(*args,**kwargs)
        if self.recording:
            self.cache.setdefault(self.key,{})["scales"]=output.scales.detach().clone()
        elif self.mode=="scale":
            output.scales=self.cache[self.key]["scales"]
            extrinsics=args[0] if args else kwargs["extrinsics"]
            rotation=extrinsics[..., :3,:3]
            output.covariances=rotation@build_covariance(output.scales,output.rotations)@rotation.transpose(-1,-2)
            assert not output.scales.requires_grad
            self.max_error=max(self.max_error,float((output.scales-self.cache[self.key]["scales"]).abs().max()))
            self.calls+=1
        return output


def save_image(x, path):
    x = x.detach().cpu().float()
    if x.ndim == 2:
        x = x[None].repeat(3, 1, 1)
    Image.fromarray((x.clamp(0, 1).permute(1, 2, 0).numpy()*255).astype(np.uint8)).save(path)


class Data:
    def __init__(self, dynamic_training=False):
        self.dynamic_training = dynamic_training
        self.root = ROOT / "datasets/nuscenes"
        self.nusc = NuScenes(version="v1.0-mini", dataroot=str(self.root), verbose=False)
        self.cache = {}
        self.dynamic_tokens = {a["token"] for a in self.nusc.attribute if a["name"] in {"vehicle.moving", "pedestrian.moving", "cycle.with_rider"}}

    def pose(self, sd):
        def matrix(record):
            t = np.eye(4)
            t[:3, :3] = Quaternion(record["rotation"]).rotation_matrix
            t[:3, 3] = record["translation"]
            return t
        return matrix(self.nusc.get("ego_pose", sd["ego_pose_token"])) @ matrix(self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"]))

    def camera(self, token):
        if token in self.cache:
            return self.cache[token]
        sample = self.nusc.get("sample", token)
        sd = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        im = torch.tensor(np.array(Image.open(self.root / sd["filename"]).convert("RGB")), dtype=torch.float32).permute(2,0,1)/255
        k = torch.tensor(self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])["camera_intrinsic"],dtype=torch.float32)
        k[0] /= im.shape[2]; k[1] /= im.shape[1]
        im,k = rescale_and_crop(im,k,(256,256))
        pose = self.pose(sd)
        # Conservative annotation-only exclusion of ALL vehicles, people and cycles.
        # No point clouds or LiDAR depths are read. Mask affects diagnostics only.
        mask_image = Image.new("L", (256,256), 255)
        draw = ImageDraw.Draw(mask_image)
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            if self.dynamic_training:
                selected = bool(set(ann["attribute_tokens"]) & self.dynamic_tokens)
            else:
                selected = ann["category_name"].startswith(("vehicle", "human"))
            if not selected:
                continue
            corners = self.nusc.get_box(ann_token).corners()
            xyz = np.linalg.inv(pose) @ np.concatenate([corners,np.ones((1,8))])
            xyz = xyz[:3,xyz[2]>(.01 if self.dynamic_training else .1)]
            if xyz.shape[1] == 0: continue
            uv = k.numpy() @ xyz; uv = 256*uv[:2]/uv[2:3]
            low,high=uv.min(1),uv.max(1)
            if self.dynamic_training:
                if high[0]<0 or low[0]>=256 or high[1]<0 or low[1]>=256:
                    continue
                pad=np.maximum(8,np.round((high-low)*.15))
                lo=np.maximum(np.floor(low-pad),0); hi=np.minimum(np.ceil(high+pad),255)
            else:
                lo = np.maximum(low-8,0); hi = np.minimum(high+8,255)
            if (hi>=lo).all(): draw.rectangle([*lo,*hi], fill=0)
        static = torch.tensor(np.array(mask_image)>0)
        result = (im,k,torch.tensor(pose,dtype=torch.float64),static)
        self.cache[token]=result
        return result

    def candidates(self):
        rows=[]
        for scene in sorted(self.nusc.scene,key=lambda s:s["name"]):
            chain=[]; token=scene["first_sample_token"]
            while token:
                sample=self.nusc.get("sample",token); chain.append(sample); token=sample["next"]
            for center in range(1,len(chain)-1,3):
                tokens=[chain[center-1]["token"],chain[center+1]["token"],chain[center]["token"]]
                sds=[self.nusc.get("sample_data",s["data"]["CAM_FRONT"]) for s in [chain[center-1],chain[center+1]]]
                rel=np.linalg.inv(self.pose(sds[0]))@self.pose(sds[1]); t=rel[:3,3]
                baseline=float(np.linalg.norm(t))
                if baseline<1: continue
                lateral=float(np.linalg.norm(t[:2])); ratio=lateral/baseline
                im,k,_,mask=self.camera(tokens[2])
                texture=float((im[:,:,1:]-im[:,:,:-1]).abs().mean())
                # Depth sensitivity for central 16x16 feature pixels, both directions.
                yy,xx=np.mgrid[24:40,24:40]; p=np.stack([xx.ravel(),yy.ravel(),np.ones(256)])
                kk=k.numpy().copy(); kk[:2]*=64
                rays=np.linalg.inv(kk)@p
                sens=[]
                for transform in [rel,np.linalg.inv(rel)]:
                    uvs=[]
                    for z in [30,50]:
                        q=kk@(transform[:3,:3]@rays*z+transform[:3,3,None]); uvs.append(q[:2]/q[2:3])
                    sens.append(float(np.median(np.linalg.norm(uvs[0]-uvs[1],axis=0))))
                rows.append(dict(scene=scene["name"],tokens=tokens,baseline=baseline,lateral=lateral,lateral_ratio=ratio,sensitivity=float(np.mean(sens)),texture=texture,static_fraction=float(mask.float().mean())))
        return rows

    def batch(self,row):
        cams=[self.camera(t) for t in row["tokens"]]
        # Recenter world coordinates without changing relative geometry.
        origin=cams[2][2][:3,3].clone()
        poses=torch.stack([c[2] for c in cams]); poses[:,:3,3]-=origin
        views={"image":torch.stack([c[0] for c in cams]),"intrinsics":torch.stack([c[1] for c in cams]),"extrinsics":poses.float(),"near":torch.ones(3),"far":torch.full((3,),100.)}
        result={"context":{k:v[:2][None].cuda() for k,v in views.items()},"target":{k:v[2:][None].cuda() for k,v in views.items()},"static":torch.stack([c[3] for c in cams]).cuda()}
        result["context"]["_sample_key"]=tuple(row["tokens"][:2])
        return result


def render(g,b,alpha=False):
    t=b["target"]
    return render_cuda(t["extrinsics"][:,0],t["intrinsics"][:,0],t["near"][:,0],t["far"][:,0],(256,256),torch.zeros(1,3,device="cuda"),g.means,g.covariances,torch.ones_like(g.harmonics[...,:1]) if alpha else g.harmonics,g.opacities,use_sh=not alpha)


def subset(g,start,end):
    return Gaussians(**{f.name:getattr(g,f.name)[:,start:end] for f in fields(g)})


def gradient(im):
    gray=im.mean(0)[None,None]
    k=torch.tensor([[-1.,0,1],[-2,0,2],[-1,0,1]],device=im.device)[None,None]/8
    return (F.conv2d(gray,k,padding=1).square()+F.conv2d(gray,k.transpose(-1,-2),padding=1).square()).sqrt()[0,0]


def projected_sizes(g,b):
    w2c=b["target"]["extrinsics"][0,0].inverse(); r=w2c[:3,:3]
    p=g.means[0]@r.T+w2c[:3,3]; z=p[:,2].clamp_min(.01)
    k=b["target"]["intrinsics"][0,0].clone()
    k[:2]*=256
    j=torch.zeros(len(p),2,3,device=p.device)
    j[:,0,0]=k[0,0]/z; j[:,0,2]=-k[0,0]*p[:,0]/z.square()
    j[:,1,1]=k[1,1]/z; j[:,1,2]=-k[1,1]*p[:,1]/z.square()
    cov=r@g.covariances[0]@r.T
    eig=torch.linalg.eigvalsh(j@cov@j.transpose(-1,-2)).clamp_min(0)
    radii=3*eig[:,-1].sqrt()  # geometric 3-sigma radius, not rasterizer's exact radius
    uv=p@k.T; uv=uv[:,:2]/uv[:,2:3]
    valid=(p[:,2]>1)&(p[:,2]<100)&(uv>=0).all(1)&(uv<256).all(1)&(g.opacities[0]>.05)
    pix=uv.nan_to_num().long().clamp(0,255)
    valid &= b["static"][2,pix[:,1],pix[:,0]]
    return radii,valid


def evaluate(model,rows,data,group,step,hook,baselines,metrics):
    model.eval()
    for i,row in enumerate(rows):
        b=data.batch(row); dump={}; folder=OUT/group/f"step_{step:04d}"/f"sample_{i}"; folder.mkdir(parents=True,exist_ok=True)
        with torch.no_grad():
            g=model(b["context"],step,visualization_dump=dump)
            depth=dump["depth"].reshape(2,256,256); residual=hook["head"][:,0]
            rgb=render(g,b)[0]; alpha=render(g,b,True)[0,0]
            n=g.means.shape[1]//2
            rgbs=[render(subset(g,s*n,(s+1)*n),b)[0] for s in range(2)]
            alphas=[render(subset(g,s*n,(s+1)*n),b,True)[0,0] for s in range(2)]
            static=b["static"][2]
            overlap=static&(alphas[0]>.5)&(alphas[1]>.5)
            radii,visible=projected_sizes(g,b)
            key=(group,i)
            if step==0:
                baselines[key]={"depth":depth.cpu(),"residual":residual.cpu(),"overlap":overlap.cpu()}
            common=overlap&baselines[key]["overlap"].cuda()
            target=b["target"]["image"][0,0]
            def mean(x,mask): return float(x[mask].mean()) if mask.any() else None
            gs=[gradient(x) for x in rgbs]; gt=gradient(target); gr=gradient(rgb)
            edges=[x>torch.quantile(x[static],.8) for x in gs]
            # Symmetric nearest-edge distance, restricted to common static alpha support.
            from scipy.ndimage import distance_transform_edt
            cm=common.cpu().numpy(); e=[(x&common).cpu().numpy() for x in edges]
            distances=[]
            if e[0].any() and e[1].any():
                distances=[distance_transform_edt(~e[1])[e[0]].mean(),distance_transform_edt(~e[0])[e[1]].mean()]
            base=baselines[key]
            delta=(depth.cpu()-base["depth"]).abs().cuda()
            sm=b["static"][:2]
            central=torch.zeros_like(sm); central[:,96:160,96:160]=True
            record=dict(group=group,step=step,sample=i,scene=row["scene"],common_pixels=int(common.sum()),static_mse=mean((rgb-target).square().mean(0),static),gradient_ratio=mean(gr,common)/max(mean(gt,common),1e-8) if common.any() else None,gradient_error=mean((gr-gt).abs(),common),split_rgb_difference=mean((rgbs[0]-rgbs[1]).abs().mean(0),common),split_edge_distance_px=float(np.mean(distances)) if distances else None,alpha=mean(alpha,static),depth_change_center=mean(delta,sm&central),depth_change_outer=mean(delta,sm&~central),residual_change=float((residual.cpu()-base["residual"]).abs().mean()),projected_radius_median=float(radii[visible].median()) if visible.any() else None,opacity_mean=float(g.opacities.mean()))
            metrics.append(record)
            for name,img in [("target",target),("combined",rgb),("context0_render",rgbs[0]),("context1_render",rgbs[1]),("alpha",alpha),("alpha0",alphas[0]),("alpha1",alphas[1]),("static_overlap",common.float())]: save_image(img,folder/f"{name}.png")
            np.savez_compressed(folder/"diagnostics.npz",depth=depth.cpu().numpy(),disparity_residual=residual.cpu().numpy(),projected_radius=radii.cpu().numpy(),radius_valid=visible.cpu().numpy(),opacity=g.opacities.cpu().numpy(),static_masks=b["static"].cpu().numpy())
            print(json.dumps(record),flush=True)
    (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2))
    model.train()


def main():
    global OUT
    parser=argparse.ArgumentParser(); parser.add_argument("--steps",type=int,default=100); parser.add_argument("--prepare-only",action="store_true"); parser.add_argument("--dynamic-training",action="store_true"); parser.add_argument("--clamp",choices=["depth","scale"]); args=parser.parse_args()
    if args.clamp: args.dynamic_training=True
    if args.dynamic_training: OUT=ROOT/"outputs/foe_training_dynamic"
    if args.clamp: OUT=ROOT/f"outputs/foe_clamp_{args.clamp}"
    torch.set_num_threads(4); np.random.seed(42); torch.manual_seed(42)
    OUT.mkdir(exist_ok=True,parents=True)
    data=Data(args.dynamic_training); rows=data.candidates()
    (OUT/"candidates.json").write_text(json.dumps(rows,indent=2))
    # Disjoint triplets. Pair groups by baseline, target texture and static coverage.
    a=[r for r in rows if r["lateral_ratio"]<.10 and r["sensitivity"]<.5 and r["static_fraction"]>.55]
    b=[r for r in rows if r["lateral_ratio"]>.20 and r["sensitivity"]>.5 and r["static_fraction"]>.55]
    used=set(); pairs=[]
    options=[]
    for aa in a:
        for bb in b:
            cost=abs(np.log(aa["baseline"]/bb["baseline"]))+abs(np.log(max(aa["texture"],1e-6)/max(bb["texture"],1e-6)))+abs(aa["static_fraction"]-bb["static_fraction"])
            options.append((cost,aa,bb))
    for cost,aa,bb in sorted(options,key=lambda x:x[0]):
        tokens=set(aa["tokens"]+bb["tokens"])
        if tokens&used or len(tokens)<6: continue
        pairs.append(dict(A=aa,B=bb,matching_cost=cost)); used|=tokens
        if len(pairs)==6: break
    if args.dynamic_training:
        pairs=json.loads((ROOT/"outputs/foe_training/selection.json").read_text())
    (OUT/"selection.json").write_text(json.dumps(pairs,indent=2))
    print(f"Candidates {len(rows)} A {len(a)} B {len(b)} matched {len(pairs)}",flush=True)
    if len(pairs)<4: raise RuntimeError("Insufficient disjoint groups; do not silently weaken selection.")
    if args.prepare_only: return
    assert torch.cuda.is_available(), "CUDA required"
    base=OmegaConf.load(ORIGINAL/"config/model/encoder/costvolume.yaml")
    exp=OmegaConf.load(ORIGINAL/"config/experiment/re10k.yaml")
    cfg=OmegaConf.merge(base,exp.model.encoder,{"unimatch_weights_path":None})
    set_cfg(OmegaConf.create({"mode":"test","dataset":{"view_sampler":{"num_context_views":2}}}))
    checkpoint=torch.load(ROOT/"checkpoints/re10k.ckpt",map_location="cpu",weights_only=False)
    weights={k[len("encoder."):]:v for k,v in checkpoint["state_dict"].items() if k.startswith("encoder.")}
    del checkpoint
    perceptual=LPIPS(net="vgg").cuda().eval(); perceptual.requires_grad_(False)
    metrics=[]; baselines={}
    settings=dict(source_commit="660f49c",steps=args.steps,seed=42,batch_size=1,loss="MSE + 0.05 LPIPS(VGG), all pixels",lidar=False,all_encoder_parameters_trainable=True,near=1,far=100,resolution=[256,256],optimizer="Adam lr=2e-4; original 300001-step cosine schedule, first pilot steps",diagnostic_steps=[0,10,50,args.steps],limitations=["Small matched observational pilot, one seed", "Depth distribution not matched by ground truth; no LiDAR", "Annotation masks for diagnostics only, conservative movable classes", "Train/eval tokens disjoint but scene overlap possible", "Projection radius is geometric 3-sigma, not exact rasterizer footprint", "Batch size reduced to 1; not full original training reproduction"])
    if args.dynamic_training:
        settings.update(loss="static normalized MSE + 0.05 LPIPS on identical neutral-filled dynamic regions",dynamic_mask="vehicle.moving/pedestrian.moving/cycle.with_rider; bbox expansion 15%, minimum 8px",selection="exact original triplets reused",limitations=["Context RGB remains intact; masks remove direct dynamic target supervision, not all indirect dynamic influence", "LPIPS has spatial receptive fields; masking is not pure static-feature LPIPS", "Small single-seed pilot; scene and true-depth confounds remain", "CUDA rasterization may be nondeterministic"])
        audit=[]
        for i,pair in enumerate(pairs):
            for group in ["A","B"]:
                for view,token in enumerate(pair[group]["tokens"]):
                    im,_,_,static=data.camera(token)
                    folder=OUT/"masks"; folder.mkdir(exist_ok=True)
                    save_image(static.float(),folder/f"{group}_{i}_view{view}_static.png")
                    overlay=im.clone(); overlay[0,~static]=1; overlay[1:,~static]*=.3
                    save_image(overlay,folder/f"{group}_{i}_view{view}_overlay.png")
                    audit.append(dict(group=group,pair=i,view=view,token=token,dynamic_fraction=float((~static).float().mean())))
        (OUT/"mask_audit.json").write_text(json.dumps(audit,indent=2))
    settings["output_clamp"]=args.clamp
    (OUT/"settings.json").write_text(json.dumps(settings,indent=2)); OmegaConf.save(cfg,OUT/"encoder.yaml")
    for group in ["A","B"]:
        torch.manual_seed(42)
        model=EncoderCostVolume(from_dict(EncoderCostVolumeCfg,OmegaConf.to_container(cfg))).cuda()
        model.load_state_dict(weights,strict=True)
        hook={}
        handle=model.depth_predictor.to_disparity.register_forward_hook(lambda m,i,o:hook.update(head=o.detach()))
        optimizer=torch.optim.Adam(model.parameters(),lr=2e-4)
        scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,2e-4,300011,pct_start=.01,cycle_momentum=False,anneal_strategy="cos")
        train_rows=[p[group] for p in pairs[:-2]]; eval_rows=[p[group] for p in pairs[-2:]]
        clamp=None
        if args.clamp:
            clamp=OutputClamp(model,args.clamp)
            model.eval()
            with torch.no_grad():
                for row in train_rows+eval_rows: model(data.batch(row)["context"],0)
            clamp.recording=False
            print(f"CLAMP {group} {args.clamp}: cached {len(clamp.cache)} triplets",flush=True)
        evaluate(model,eval_rows,data,group,0,hook,baselines,metrics)
        history=[]
        for step in range(1,args.steps+1):
            batch=data.batch(train_rows[(step-1)%len(train_rows)])
            optimizer.zero_grad(set_to_none=True)
            gaussians=model(batch["context"],step)
            predicted=render(gaussians,batch); target=batch["target"]["image"][:,0]
            if args.dynamic_training:
                valid=batch["static"][2][None,None].to(predicted.dtype)
                if valid.sum()==0: raise RuntimeError("No static target pixels")
                mse=((predicted-target).square()*valid).sum()/(3*valid.sum())
                # Identical neutral fills: dynamic RGB contributes neither mismatch nor gradient.
                pred_static=predicted*valid+.5*(1-valid)
                target_static=target*valid+.5*(1-valid)
                lp=perceptual(pred_static,target_static,normalize=True).mean()
            else:
                mse=(predicted-target).square().mean(); lp=perceptual(predicted,target,normalize=True).mean()
            loss=mse+.05*lp
            if args.dynamic_training and step<=len(train_rows):
                grad=torch.autograd.grad(loss,predicted,retain_graph=True)[0]
                assert torch.count_nonzero(grad*(1-valid))==0, "Dynamic target gradient must vanish"
                print(f"MASK_GRADIENT_CHECK {group} step={step} dynamic_fraction={float(1-valid.mean()):.5f} dynamic_gradient=0",flush=True)
            if not torch.isfinite(loss): raise RuntimeError("Non-finite loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),.5); optimizer.step(); scheduler.step()
            history.append(dict(step=step,mse=float(mse),lpips=float(lp),loss=float(loss),lr=optimizer.param_groups[0]["lr"]))
            if step%10==0: print(group,history[-1],flush=True)
            del gaussians,predicted,loss,mse,lp
            if step in {10,50,args.steps}:
                evaluate(model,eval_rows,data,group,step,hook,baselines,metrics)
        torch.save({"state_dict":model.cpu().state_dict(),"settings":settings},OUT/f"{group}_final.pt")
        (OUT/f"{group}_training.json").write_text(json.dumps(history,indent=2))
        if clamp:
            audit=dict(mode=args.clamp,cached_inputs=len(clamp.cache),replacement_calls=clamp.calls,max_output_error=clamp.max_error)
            (OUT/f"{group}_clamp_audit.json").write_text(json.dumps(audit,indent=2))
            torch.save({str(k):{name:v.cpu() for name,v in values.items()} for k,values in clamp.cache.items()},OUT/f"{group}_initial_outputs.pt")
            print("CLAMP_AUDIT",group,audit,flush=True)
            clamp.pre.remove(); clamp.depth_hook.remove()
            model.gaussian_adapter.forward=clamp.original_adapter
            del clamp
        handle.remove(); del model,optimizer,scheduler; torch.cuda.empty_cache()


if __name__=="__main__": main()
