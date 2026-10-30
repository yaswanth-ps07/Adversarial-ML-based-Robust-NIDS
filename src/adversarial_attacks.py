import torch
import torch.nn.functional as F
from config import CLIP_MIN,CLIP_MAX

def fgsm(model,x,y,epsilon=0.1):
    model.eval(); x0=x.detach(); xa=x0.clone().requires_grad_(True)
    loss=F.binary_cross_entropy_with_logits(model(xa),y.float()); model.zero_grad(); loss.backward()
    adv=xa+epsilon*xa.grad.sign()
    return torch.clamp(adv,CLIP_MIN,CLIP_MAX).detach()

def pgd(model,x,y,epsilon=0.1,alpha=0.02,steps=7):
    model.eval(); x0=x.detach(); adv=x0.clone()
    for _ in range(steps):
        adv.requires_grad_(True); loss=F.binary_cross_entropy_with_logits(model(adv),y.float()); model.zero_grad(); loss.backward()
        adv=adv.detach()+alpha*adv.grad.sign(); delta=torch.clamp(adv-x0,-epsilon,epsilon); adv=torch.clamp(x0+delta,CLIP_MIN,CLIP_MAX).detach()
    return adv
