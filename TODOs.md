# TODOs (Project Handoff)

This file summarizes *larger, unfinished concepts* identified in the repo. It focuses on feature gaps and unsupported operations rather than small inline TODOs.

## Sampling & Differentiable Routing Gaps
- SOCS: no EM, no MPE, no conditional sampling with evidence, no differentiable routing, and only scalar output supported (spflow/modules/sos/socs.py).
- SignedSum: no EM, no marginalize, no differentiable routing, no conditional sampling; sampling only when all weights are non-negative and repetitions==1 (spflow/modules/sums/signed_sum.py).
- CLTree: differentiable routing unsupported (spflow/modules/leaves/cltree.py).
- PiecewiseLinear: differentiable sampling unsupported; MLE not supported (initialize-only) (spflow/modules/leaves/piecewise_linear.py).
- ElementwiseSum + Cat: differentiable sampling requires hard=True (soft routing not supported) (spflow/modules/sums/elementwise_sum.py, spflow/modules/ops/cat.py).
- SignedCategorical + ExpSOCS: sampling unsupported; SignedCategorical also disallows NaN evidence; EM unsupported (spflow/zoo/sos/signed_categorical.py, spflow/zoo/sos/exp_socs.py).

## SamplingContext Repetition-Index Invariant
- `SamplingContext.repetition_index` is a required runtime invariant for all repetition counts, including single-repetition modules. Missing repetition indices are rejected at assignment and validation boundaries; constructor omission still initializes explicit defaults.

## Einet Sampling Limitation
- Bottom-up Einet structure cannot sample yet; only top-down sampling is implemented (spflow/zoo/einet/einet.py).
