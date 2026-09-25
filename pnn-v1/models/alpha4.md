# PNN-v1 Alpha4

**Status:** Active PNN-v1 architecture  
**Promoted by:** PNN-v1 Exp011  
**Domain tested:** ECG200 time-series classification  
**Backend:** Q.ANT CPU/software simulation

## Architecture

```
96 input samples
  ↓
6 contiguous windows × 16 samples
  ↓
6 × Q.ANT Fourier/KAN blocks: 16 → 16
  ↓
concatenate → 96 features
  ↓
Q.ANT Fourier/KAN head: 96 → 2
  ↓
2 class logits
```

Every Fourier/KAN block uses the compact frequency basis:

```
k = [1, 2]
```

**Parameter count:** 7,010.

## Confirmatory result

In Exp011, Alpha4 was trained on all 100 canonical ECG200 TRAIN cases for each of ten frozen seeds and evaluated on the canonical 100-case TEST split.

- Mean Q.ANT TEST accuracy: **89.3%**
- Across-seed standard deviation: **1.42 percentage points**
- Aggregate correct predictions: **893 / 1,000**
- Mean reference accuracy: **89.6%**
- Mean reference/Q.ANT prediction disagreements: **0.3 per 100 predictions**
- Mean absolute logit error: **0.03264**

The previous active Alpha3 reached 85.2% Q.ANT TEST accuracy and 852 / 1,000 correct predictions under the same confirmation protocol. Alpha4 therefore gained 41 correct classifications and satisfied the preregistered promotion criterion.

## Research meaning

Alpha4 combines two architecture signals discovered by the research loop: a compact k=[1,2] Fourier basis and a larger 16-sample local temporal receptive field.

This is an ECG200 result, not evidence of general superiority. The next milestone is transfer validation on another time-series dataset.

No physical photonic hardware performance is measured here.
