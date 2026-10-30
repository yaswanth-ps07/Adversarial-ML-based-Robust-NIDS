import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report

def calculate_metrics(y_true,y_pred,clean_pred=None):
    cm=confusion_matrix(y_true,y_pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    out={'accuracy':accuracy_score(y_true,y_pred),'precision':precision_score(y_true,y_pred,zero_division=0),'recall':recall_score(y_true,y_pred,zero_division=0),'f1':f1_score(y_true,y_pred,zero_division=0),'false_positive_rate':fp/(fp+tn) if fp+tn else 0.0}
    if clean_pred is not None:
        initially_correct=(np.asarray(clean_pred)==np.asarray(y_true))
        out['attack_success_rate']=float(np.mean(np.asarray(y_pred)[initially_correct]!=np.asarray(y_true)[initially_correct])) if initially_correct.any() else 0.0
    else: out['attack_success_rate']=None
    return out,cm,classification_report(y_true,y_pred,target_names=['NORMAL','ATTACK'],zero_division=0)
