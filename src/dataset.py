from pathlib import Path
import pandas as pd

def find_csv_files(data_dir):
    files = sorted(Path(data_dir).rglob('*.csv'))
    return [p for p in files if not p.name.startswith('.')]

def load_cicids2017(data_dir, max_rows=None, seed=42):
    files = find_csv_files(data_dir)
    if not files:
        raise FileNotFoundError(f'No CSV files found in {data_dir}. See data/README.md.')
    frames = []
    for path in files:
        df = pd.read_csv(path, low_memory=False)
        df['__source_file'] = path.name
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    if max_rows and len(data) > max_rows:
        label_col = next((c for c in data.columns if c.strip().lower() == 'label'), None)
        if label_col:
            groups=[]
            for _, g in data.groupby(label_col):
                n=max(1, round(max_rows*len(g)/len(data)))
                groups.append(g.sample(min(n,len(g)), random_state=seed))
            data=pd.concat(groups,ignore_index=True).sample(frac=1,random_state=seed).head(max_rows)
        else:
            data=data.sample(max_rows, random_state=seed)
    return data.reset_index(drop=True), files

def validate_dataset(df):
    label_cols=[c for c in df.columns if c.strip().lower()=='label']
    if not label_cols:
        raise ValueError('CIC-IDS2017 CSVs must contain a Label column.')
    if len(df)<100:
        raise ValueError('Dataset contains too few rows for training.')
    return label_cols[0]
