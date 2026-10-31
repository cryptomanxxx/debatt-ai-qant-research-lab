# CL-2 matched local hidden-layer rule — engineering candidate

**Status:** no dataset experiment, not scientifically approved or executed.

`LocalHebbianMLP` inherits the same 16-input/8-tanh-hidden/2-output architecture and initialization as `TinyDenseMLP` (**154 trainable parameters including biases**). The critical intervention is an ablation of the learning rule:

- Dense baseline: backpropagated supervised gradients train both layers by full-parameter SGD.
- Local candidate: hidden layer weights update from *only active input indices* and the currently computed postsynaptic hidden value; **no target-label gradient is propagated to the hidden layer**. The output readout still uses a globally computed softmax error and externally supplied class label, so this is a **hybrid rule, not fully local or biologically faithful STDP**.
- First-layer inactive input weights are untouched during local steps; the readout and bias updates remain dense. The corrected `Counters` cover **all model-weight reads and writes in the current dense forward and update loops** (bias operations are reported separately by `cl2_operation_accounting.StepProfile`). The local forward path is **still dense**. These logical source-level counts exclude encoding, scanning for active indices, nonlinear operations, optimizer control and physical hardware traffic. They **must not** be interpreted as actual memory reads/writes, speed, watts or joules.
- Both models use identical initial weights with identical seed, but training algorithms differ, so equal parameter counts alone do **not** establish matched effective capacity or task quality.

Contracts check initial-parameter equality, hidden-update label independence, inactive-index preservation, counted operations and validation. One local update on a synthetic fixed input is a **unit test**, not a continual-learning experimental run.

**Before scientific claims:** freeze the shared-rule dataset candidate and protocol, check realized input activity, choose paired training budgets/hyperparameters, compare at **matched predictive quality**, include baseline majority and oracle references, validate separately across seeds, and obtain explicit approval for bounded training compute. A future *temporal* SNN and calibrated hardware-memory model remain necessary for the broader neuromorphic/fotoniska hypothesis.
