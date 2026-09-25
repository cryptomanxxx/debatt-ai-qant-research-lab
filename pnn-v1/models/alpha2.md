# PNN-v1 Alpha2

**Status:** active architecture  
**Promoted by:** PNN-v1 Exp007  
**Backend validated:** Q.ANT CPU/software simulation  
**Promotion task:** ECG200 binary time-series classification

## Architecture

96 ECG samples → 12 contiguous windows × 8 samples → 12 × Q.ANT Fourier/KAN 8→8 → concatenate to 96 learned features → Q.ANT Fourier/KAN 96→2 → 2 logits.

Frequency basis: k=[1,2,3,4]. Parameter count: 7,778. Promotion experiment used raw ECG200 values.

Training uses the exact PyTorch cosine formulation; Q.ANT evaluation uses qant.ai.calc_kan_layer_fprop with bias through qant.ai.add_bias_fprop.

## Promotion evidence
Exp005 selected 8→8 using frozen TRAIN-only validation. Exp006 found mean Q.ANT accuracy 0.834 versus Alpha1 0.824 with lower variation, but hit a floating-point boundary in the decision expression. Exp007 used the preregistered integer-count gate: Alpha1 824/1,000 correct; Alpha2 834/1,000; difference +10. Both frozen criteria were met.

## Scope
Alpha2 is a research architecture, not a production model. Evidence is currently ECG200-specific. The Exp006 MLP baseline achieved 0.876 mean accuracy, so Alpha2 has not matched it. Results use Q.ANT software/simulation CPU backend and do not measure physical photonic hardware performance.

## Research rule
Future architecture experiments compare against this frozen Alpha2 definition unless a later preregistered experiment explicitly promotes a successor.
