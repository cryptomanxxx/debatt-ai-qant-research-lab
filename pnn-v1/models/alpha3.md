# PNN-v1 Alpha3

**Status:** active architecture  
**Promoted by:** PNN-v1 Exp009  
**Backend validated:** Q.ANT CPU/software simulation  
**Promotion task:** ECG200 binary time-series classification

## Architecture

96 ECG samples → 12 contiguous windows × 8 samples → 12 × Q.ANT Fourier/KAN 8→8 → concatenate to 96 learned features → Q.ANT Fourier/KAN 96→2 → 2 logits.

**Frequency basis:** k=[1,2]  
**Parameter count:** 3,938

The topology is unchanged from Alpha2; Alpha3 replaces the four-term k=[1,2,3,4] basis with the experimentally confirmed compact two-term basis.

## Promotion evidence
Exp008 discovered k=[1,2] as a promising TRAIN-validation candidate but did not promote it because it failed the preregistered stability gate. Exp009 froze that candidate and performed a full-TRAIN confirmation. Across 10 seeds × 100 TEST cases, Alpha3 achieved 852 correct Q.ANT classifications versus 834 for Alpha2, mean Q.ANT accuracy 0.852 versus 0.834, and lower standard deviation 0.01661 versus 0.02154. Parameter count fell from 7,778 to 3,938.

## Current benchmark gap
The conventional MLP calibration achieved 0.876 mean accuracy with 8,354 parameters. Alpha3 therefore remains 0.024 absolute below that baseline while using fewer than half as many parameters.

## Scope
Alpha3 is a research architecture. Evidence is currently ECG200-specific. Results use Q.ANT software/simulation CPU backend and do not measure physical photonic hardware performance.

## Research rule
Future architecture experiments compare against this frozen Alpha3 definition unless a later preregistered experiment explicitly promotes a successor.
