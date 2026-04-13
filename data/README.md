# Data Format

Training and validation splits are described by CSV manifests such as `train.datalist.csv` and `val.datalist.csv`.

Each manifest must have these columns:

- `sample`: sample identifier used to pair rows
- `kind`: either `source` or `target`
- `path`: image path relative to the CSV file

Example:

```csv
sample,kind,path
image_0,source,train/source/image_0.png
image_0,target,train/target/image_0.png
image_1,source,train/source/image_1.png
image_1,target,train/target/image_1.png
```

Each sample must appear exactly twice: one `source` row and one `target`
row. All paired images must have the same resolution, and all samples in the
same split must share one resolution.
