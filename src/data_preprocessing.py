import json, re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import ARTIFACT_DIR, SEED, TEST_SIZE, VAL_SIZE

def clean_name(name):
    return re.sub(r'[^a-z0-9]+','_',str(name).strip().lower()).strip('_')

def binary_label(value):
    s=str(value).strip().lower()
    return 0 if s in {'benign','normal','0','normal.'} else 1

def prepare_dataframe(df, label_col):
    df=df.copy()
    df.columns=[clean_name(c) for c in df.columns]
    label_col=clean_name(label_col)
    df=df.drop_duplicates().reset_index(drop=True)
    y=df[label_col].map(binary_label).astype(np.int64)
    drop_cols={label_col,'source_file','flow_id','source_ip','destination_ip','src_ip','dst_ip','timestamp','simillarhttp'}
    X=df.drop(columns=[c for c in drop_cols if c in df.columns],errors='ignore')
    X=X.replace([np.inf,-np.inf],np.nan)
    for c in X.columns:
        X[c]=pd.to_numeric(X[c],errors='coerce')
    X=X.dropna(axis=1,how='all')
    if X.shape[1]==0:
        raise ValueError('No usable numeric features remained after preprocessing.')
    return X,y

def split_and_scale(X,y,save=True):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=TEST_SIZE,random_state=SEED,stratify=y)
    relative_val=VAL_SIZE/(1-TEST_SIZE)
    X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=relative_val,random_state=SEED,stratify=y_train)
    medians=X_train.median(numeric_only=True).fillna(0.0)
    X_train=X_train.fillna(medians); X_val=X_val.fillna(medians); X_test=X_test.fillna(medians)
    scaler=StandardScaler()
    Xt=scaler.fit_transform(X_train).astype('float32')
    Xv=scaler.transform(X_val).astype('float32')
    Xs=scaler.transform(X_test).astype('float32')
    if save:
        (ARTIFACT_DIR/'scaler').mkdir(parents=True,exist_ok=True)
        joblib.dump(scaler,ARTIFACT_DIR/'scaler'/'standard_scaler.joblib')
        joblib.dump(medians,ARTIFACT_DIR/'scaler'/'training_medians.joblib')
        (ARTIFACT_DIR/'feature_names.json').write_text(json.dumps(list(X.columns),indent=2))
    return Xt,Xv,Xs,y_train.to_numpy(),y_val.to_numpy(),y_test.to_numpy(),scaler
