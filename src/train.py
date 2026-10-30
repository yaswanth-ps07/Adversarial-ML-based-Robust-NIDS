import json,sys,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from config import *
from src.dataset import load_cicids2017,validate_dataset
from src.data_preprocessing import prepare_dataframe,split_and_scale
from src.model import NIDSDNN
from src.visualization import save_history

def seed_all(): random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
def loaders(X,y,shuffle): return DataLoader(TensorDataset(torch.tensor(X),torch.tensor(y,dtype=torch.float32)),batch_size=BATCH_SIZE,shuffle=shuffle)
def run_epoch(model,loader,opt=None):
    train=opt is not None; model.train(train); lossfn=nn.BCEWithLogitsLoss(); total=correct=0; losses=0
    for x,y in loader:
        if train: opt.zero_grad()
        logits=model(x); loss=lossfn(logits,y)
        if train: loss.backward(); opt.step()
        losses+=loss.item()*len(y); correct+=((torch.sigmoid(logits)>=.5)==y.bool()).sum().item(); total+=len(y)
    return losses/total,correct/total
def train_model(Xt,yt,Xv,yv,out_path,history_path):
    seed_all(); model=NIDSDNN(Xt.shape[1]); opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE); best=float('inf'); patience=4; wait=0; history=[]
    tr=loaders(Xt,yt,True); va=loaders(Xv,yv,False)
    for epoch in range(1,EPOCHS+1):
        tl,ta=run_epoch(model,tr,opt); vl,va_acc=run_epoch(model,va); history.append({'epoch':epoch,'train_loss':tl,'val_loss':vl,'train_accuracy':ta,'val_accuracy':va_acc})
        if vl<best-1e-4: best=vl; wait=0; torch.save(model.state_dict(),out_path)
        else: wait+=1
        if wait>=patience: break
    Path(history_path).write_text(json.dumps(history,indent=2)); save_history(history,FIGURE_DIR/(Path(history_path).stem+'_loss.png')); return model

def main():
    for p in [MODEL_DIR,METRIC_DIR,FIGURE_DIR]: p.mkdir(parents=True,exist_ok=True)
    df,files=load_cicids2017(DATA_DIR,MAX_ROWS,SEED); label=validate_dataset(df); X,y=prepare_dataframe(df,label); Xt,Xv,Xs,yt,yv,ys,_=split_and_scale(X,y)
    np.savez_compressed(ARTIFACT_DIR/'splits.npz',X_train=Xt,X_val=Xv,X_test=Xs,y_train=yt,y_val=yv,y_test=ys)
    metadata={'rows_loaded':len(df),'features':Xt.shape[1],'normal':int((y==0).sum()),'attack':int((y==1).sum()),'files':[p.name for p in files]}; (ARTIFACT_DIR/'dataset_metadata.json').write_text(json.dumps(metadata,indent=2))
    train_model(Xt,yt,Xv,yv,MODEL_DIR/'baseline_dnn.pt',METRIC_DIR/'baseline_history.json'); print('Baseline training complete.')
if __name__=='__main__': main()
