# CIC-IDS2017 dataset setup

Use the official Canadian Institute for Cybersecurity / University of New Brunswick CIC-IDS2017 dataset page:
https://www.unb.ca/cic/datasets/ids-2017.html

Download the **MachineLearningCSV / generated labelled flow CSV files** supplied for CIC-IDS2017. Extract the CSV files into:

`data/raw/`

The loader recursively finds `.csv` files, so keeping the original extracted subfolders is also fine. Do not commit the dataset to Git; it is intentionally excluded by `.gitignore`.

Expected data contains a `Label` column. The pipeline maps `BENIGN` to 0 and every attack label to 1.
