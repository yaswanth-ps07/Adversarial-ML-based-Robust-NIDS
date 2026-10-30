import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch,shap
from config import *
from src.model import load_model

def main():
    z=np.load(ARTIFACT_DIR/'splits.npz'); X=z['X_test']; features=json.loads((ARTIFACT_DIR/'feature_names.json').read_text()); model=load_model(MODEL_DIR/'baseline_dnn.pt',X.shape[1]); model.eval()
    class ShapWrapper(torch.nn.Module):
        def __init__(self, inner): super().__init__(); self.inner=inner
        def forward(self, x): return self.inner(x).unsqueeze(1)
    wrapped=ShapWrapper(model)
    bg=torch.tensor(X[:min(SHAP_MAX_BACKGROUND,len(X))]); samples=torch.tensor(X[:min(SHAP_MAX_SAMPLES,len(X))]); explainer=shap.GradientExplainer(wrapped,bg); values=explainer.shap_values(samples)
    arr=np.asarray(values); arr=np.squeeze(arr); importance=np.mean(np.abs(arr),axis=0); pd.DataFrame({'feature':features,'mean_abs_shap':importance}).sort_values('mean_abs_shap',ascending=False).to_csv(METRIC_DIR/'shap_feature_importance.csv',index=False); print('SHAP feature importance saved.')
if __name__=='__main__': main()
