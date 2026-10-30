import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from config import *
from src.model import load_model
from src.metrics import calculate_metrics
from src.adversarial_attacks import fgsm,pgd
from src.visualization import save_confusion,save_comparison

def predict(model,x):
    with torch.no_grad(): return (torch.sigmoid(model(torch.tensor(x,dtype=torch.float32)))>=.5).cpu().numpy().astype(int)
def attack_array(model,X,y,kind,epsilon):
    outs=[]
    for i in range(0,len(X),BATCH_SIZE):
        x=torch.tensor(X[i:i+BATCH_SIZE],dtype=torch.float32); t=torch.tensor(y[i:i+BATCH_SIZE],dtype=torch.float32)
        a=fgsm(model,x,t,epsilon) if kind=='fgsm' else pgd(model,x,t,epsilon,PGD_ALPHA,PGD_STEPS); outs.append(a.numpy())
    return np.concatenate(outs)
def evaluate_scenario(model,X,y,name,clean_pred=None):
    pred=predict(model,X); m,cm,report=calculate_metrics(y,pred,clean_pred); save_confusion(cm,name,FIGURE_DIR/f'{name.lower().replace(" ","_")}_cm.png'); (METRIC_DIR/f'{name.lower().replace(" ","_")}_report.txt').write_text(report); return {'scenario':name,**m},pred
def main():
    z=np.load(ARTIFACT_DIR/'splits.npz'); X=z['X_test']; y=z['y_test']; n=min(len(X),ATTACK_EVAL_MAX); X=X[:n]; y=y[:n]
    base=load_model(MODEL_DIR/'baseline_dnn.pt',X.shape[1]); rows=[]; r,clean=evaluate_scenario(base,X,y,'Baseline Clean'); rows.append(r)
    for kind,eps in [('FGSM',FGSM_EPSILON),('PGD',PGD_EPSILON)]:
        adv=attack_array(base,X,y,kind.lower(),eps); r,_=evaluate_scenario(base,adv,y,f'Baseline {kind}',clean); rows.append(r)
    robust_path=MODEL_DIR/'robust_dnn.pt'
    if robust_path.exists():
        robust=load_model(robust_path,X.shape[1]); r,rclean=evaluate_scenario(robust,X,y,'Robust Clean'); rows.append(r)
        for kind,eps in [('FGSM',FGSM_EPSILON),('PGD',PGD_EPSILON)]:
            adv=attack_array(robust,X,y,kind.lower(),eps); r,_=evaluate_scenario(robust,adv,y,f'Robust {kind}',rclean); rows.append(r)
    df=pd.DataFrame(rows); df.to_csv(METRIC_DIR/'comparison.csv',index=False)
    for metric in ['accuracy','f1','attack_success_rate']:
        save_comparison(df.fillna(0),metric,FIGURE_DIR/f'{metric}_comparison.png')
    print(df.to_string(index=False))
if __name__=='__main__': main()
