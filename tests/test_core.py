import numpy as np,pandas as pd,torch
from src.data_preprocessing import prepare_dataframe,split_and_scale
from src.model import NIDSDNN
from src.adversarial_attacks import fgsm,pgd

def fixture_df(n=200):
    rng=np.random.default_rng(42); return pd.DataFrame({' Flow Duration':rng.normal(size=n),'Total Fwd Packets':rng.normal(size=n),' Flow Bytes/s':rng.normal(size=n),' Label':['BENIGN']*(n//2)+['DDoS']*(n-n//2)})
def test_preprocessing_shapes():
    df=fixture_df(); X,y=prepare_dataframe(df,' Label'); a,b,c,ya,yb,yc,_=split_and_scale(X,y,save=False); assert a.shape[1]==3 and len(a)+len(b)+len(c)==len(df); assert set(np.unique(y))=={0,1}
def test_model_forward():
    m=NIDSDNN(3); assert m(torch.randn(8,3)).shape==(8,)
def test_attacks_are_bounded():
    m=NIDSDNN(3); x=torch.randn(16,3); y=torch.randint(0,2,(16,)).float(); eps=.1
    a=fgsm(m,x,y,eps); p=pgd(m,x,y,eps,.02,5); assert a.shape==x.shape and p.shape==x.shape; assert torch.max(torch.abs(p-x)).item()<=eps+1e-5
