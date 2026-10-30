from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

def save_confusion(cm,title,path):
    fig,ax=plt.subplots(figsize=(5,4)); ConfusionMatrixDisplay(cm,display_labels=['Normal','Attack']).plot(ax=ax,cmap='Blues',colorbar=False); ax.set_title(title); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)
def save_history(history,path):
    df=pd.DataFrame(history); fig,ax=plt.subplots(figsize=(7,4)); ax.plot(df['epoch'],df['train_loss'],label='Train loss'); ax.plot(df['epoch'],df['val_loss'],label='Validation loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(df['epoch'],df['train_accuracy'],label='Train accuracy'); ax.plot(df['epoch'],df['val_accuracy'],label='Validation accuracy'); ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.legend(); fig.tight_layout(); fig.savefig(str(path).replace('loss','accuracy'),dpi=150); plt.close(fig)
def save_comparison(df,metric,path):
    fig,ax=plt.subplots(figsize=(9,4)); df.set_index('scenario')[metric].plot(kind='bar',ax=ax); ax.set_ylim(0,1); ax.set_ylabel(metric.replace('_',' ').title()); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)
