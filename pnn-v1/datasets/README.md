# PNN-v1 datasets

Dataset manifests, loaders and documentation live here. Large datasets and generated caches should not be committed to GitHub.

## PNN-v1 Exp001 dataset: ECG200

**Dataset:** ECG200  
**Task:** univariate time-series classification  
**Domain:** ECG / single-heartbeat electrical activity  
**Classes:** 2 (normal heartbeat and myocardial infarction)  
**Canonical split:** 100 training cases + 100 test cases  
**Series length:** 96 samples per case  
**Source family:** UCR Time Series Classification Archive  
**Reproducible distribution:** TSML/Zenodo ECG200 record, DOI 10.5281/zenodo.11186675

ECG200 is fixed as the first practical dataset for PNN-v1 Exp001. We will preserve the archive's predefined TRAIN/TEST split rather than creating a favorable custom split.

### Retrieval

The experiment should retrieve ECG200 programmatically with `aeon.datasets.load_classification`, using separate `TRAIN` and `TEST` splits. Current aeon documentation states that `load_classification` downloads TSML classification collections from the Zenodo TSML community when the dataset is not already present locally.

Conceptual loader:

```python
from aeon.datasets import load_classification

X_train, y_train = load_classification("ECG200", split="TRAIN")
X_test, y_test = load_classification("ECG200", split="TEST")
```

The experiment implementation must record the dataset name, split sizes, shapes, class labels, aeon version and retrieval/source metadata in its result JSON.

### Scientific role

ECG200 is the first controlled practical benchmark, not evidence that PNN-v1 is generally superior for time-series analysis. Exp001 should compare a conventional baseline and the first Q.ANT-native PNN-v1 candidate on exactly the same predefined data split and evaluation metric.

### Provenance

- UCR Time Series Classification Archive 2018: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
- ECG200 Zenodo record: https://doi.org/10.5281/zenodo.11186675
- aeon loader documentation: https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.datasets.load_classification.html
