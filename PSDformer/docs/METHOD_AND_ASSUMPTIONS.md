# PSDformer: method specification and reproducibility audit

Audit date: 2026-10-11. Source: the supplied 15-page manuscript, *Long-Term Stress Conditions Assessment and Prediction of Drivers Based on a Hybrid Deep Learning Framework and Novel Sensor System*, by Qing Wang, Ke Li, Xiang Wang, and Jing Qin. Page numbers below count PDF pages from one. PDF SHA-256: `4d99cc73cdfd8f26d2a05b7c2c630b193fbe91042275bbd72842bd081ef0603b`.

The PDF text was read and pages 4-7, 9, and 11 were visually inspected, including Algorithm 1, equations, architecture figures, and result plots. The paper-linked GitHub repository was inspected read-only at the pinned commit below. No downloaded repository code was executed or copied into the independent implementation.

## 1. Bottom line and scope

A complete runnable **independent reference implementation of the stated mathematical pipeline** is feasible. Exact reproduction of the authors' experiments is not currently possible from the supplied evidence. The paper omits indispensable data, label, split, architectural, and training details; several figures contradict the equations or numerical tables; and the current official repository contains only a small incomplete scaffold.

Use Algorithm 1 and equations (1)-(19) as the primary architectural specification. Preserve the named modules and make unresolved choices configurable and clearly identified as engineering assumptions. Do not describe the resulting program as the authors' exact code, a reproduction of the reported 95.1% accuracy, or a validated stress/medical/safety assessment system.

In particular, the independent equation-based model **does not implement** the undefined pretrained branch, cross-modal matching, feature regularization, feature-fusion training procedure, or output-consistency objective drawn in Fig. 9. Nor does it reconstruct graph adjacency as drawn in Fig. 12. These are unresolved inconsistencies, not details that can be inferred reliably.

## 2. Verified official repository

- Repository: https://github.com/Wang-Qing-sys/PSDformer
- Inspected default branch: `main`.
- Pinned tip: [`701f6d61d12b97412ac570a47c1a3e8b5a2a7048`](https://github.com/Wang-Qing-sys/PSDformer/commit/701f6d61d12b97412ac570a47c1a3e8b5a2a7048), committed 2026-09-26.
- Recursive tree was returned with `truncated=false`.
- Only `main` was returned by the branches endpoint; the releases endpoint returned an empty list.
- Both README files contain only the project title.
- No dataset files, trained weights, dependency manifest, executable training pipeline, or license file occur in the current recursive tree. Repository metadata also reports no license. Public availability alone should not be taken as a license to redistribute its code.

### Current files and what they actually establish

All links below are pinned to the inspected commit.

| Source file | Verified behavior | Limit relative to paper |
|---|---|---|
| [config.yaml](https://github.com/Wang-Qing-sys/PSDformer/blob/701f6d61d12b97412ac570a47c1a3e8b5a2a7048/PSDformer/config.yaml) | `channels=5`, `segment_length=24`, `embed_dim=128`, `layers=3`, `prediction_length=24`; training batch 32, lr .001, Adam | Main model never loads this config; layer count and prediction length are not wired into the model |
| [models/PSDformer.py](https://github.com/Wang-Qing-sys/PSDformer/blob/701f6d61d12b97412ac570a47c1a3e8b5a2a7048/PSDformer/models/PSDformer.py) | MDSW with segment length 24 and embedding 128; flatten segment and variable axes; one TDE; scalar linear head | Does not import or call EDM or DGC; no future-query decoder or explicit forecast-horizon mapping |
| [MDSW/mdsw.py](https://github.com/Wang-Qing-sys/PSDformer/blob/701f6d61d12b97412ac570a47c1a3e8b5a2a7048/PSDformer/models/MDSW/mdsw.py) | Per-variable unfold with stride equal to segment size, shared linear segment projection; returns `[B,L,D,E]` | No positional encoding, tail-padding policy, or explicit shape validation |
| [TDE/tde.py](https://github.com/Wang-Qing-sys/PSDformer/blob/701f6d61d12b97412ac570a47c1a3e8b5a2a7048/PSDformer/models/TDE/tde.py) | One 4-head self-attention operation, residual, LayerNorm | Runs on flattened tokens; no separate temporal/dimensional axes, routers, or FFN |
| [EDM/edm.py](https://github.com/Wang-Qing-sys/PSDformer/blob/701f6d61d12b97412ac570a47c1a3e8b5a2a7048/PSDformer/models/EDM/edm.py) | Two same-width Conv1d layers followed by a linear map | Unused; no pair merging, multiscale attention decoder, or scale-wise prediction sum |
| [DGC/dgc.py](https://github.com/Wang-Qing-sys/PSDformer/blob/701f6d61d12b97412ac570a47c1a3e8b5a2a7048/PSDformer/models/DGC/dgc.py) | Single kernel-3 Conv1d with padding 1 and output width `dim+growth`; default growth 32 | Unused; no two dense blocks, pooling, or graph operator |
| [head/prediction_head.py](https://github.com/Wang-Qing-sys/PSDformer/blob/701f6d61d12b97412ac570a47c1a3e8b5a2a7048/PSDformer/models/head/prediction_head.py) | Last-axis linear projection, default one output | Produces one scalar per input token, not an explicit future stress sequence |
| [losses/loss.py](https://github.com/Wang-Qing-sys/PSDformer/blob/701f6d61d12b97412ac570a47c1a3e8b5a2a7048/PSDformer/losses/loss.py) | Mean squared error through PyTorch `mse_loss` | Supports regression MSE as a repository choice; no classification or auxiliary losses |
| [datasets/driver_dataset.py](https://github.com/Wang-Qing-sys/PSDformer/blob/701f6d61d12b97412ac570a47c1a3e8b5a2a7048/PSDformer/datasets/driver_dataset.py) | Wraps already-provided data and labels | No loader, label definition, preprocessing, windows, or splitting |
| [training/train.py](https://github.com/Wang-Qing-sys/PSDformer/blob/701f6d61d12b97412ac570a47c1a3e8b5a2a7048/PSDformer/training/train.py) | Instantiates and prints a model | No optimizer, epochs, data, validation, checkpoint, or training step |
| [evaluation/metrics.py](https://github.com/Wang-Qing-sys/PSDformer/blob/701f6d61d12b97412ac570a47c1a3e8b5a2a7048/PSDformer/evaluation/metrics.py) | Wraps sklearn MSE | No other paper metrics |

For the main scaffold, a divisible input `[B,T,D]` yields `[B,(T/24)*D,1]`. At the paper lookback `T=24` and repository default `D=5`, this is `[B,5,1]`, not `[B,24,1]`. The repository's `prediction_length=24` setting does not fix this. Combining the paper lookback with the scaffold segment size also yields only one temporal segment, so the described temporal hierarchy becomes degenerate.

### Historical file caveat

The repository history includes a 1,116-line `psdformer.py` that was added and then [deleted](https://github.com/Wang-Qing-sys/PSDformer/commit/f40ea4dea053918835a748a2a7c709f30cf7cd27) on 2026-09-26. Its introductory documentation explicitly described it as a manuscript reproduction using configurable engineering completions for unspecified hyperparameters and losses. It is absent from current `main`. Its chosen auxiliary losses and constants are therefore **not** reliable evidence of the reported experiment settings or a reason to silently reinstate that withdrawn implementation. This audit used its deletion diff only to establish provenance.

## 3. Notation and tensor contract

The paper does not give a complete end-to-end tensor shape specification. The following separates necessary consequences of its equations from choices required to make them executable.

| Symbol | Meaning |
|---|---|
| `B` | Batch size |
| `T` | Observed lookback length, measured in processed data steps |
| `D` | Number of input variables/channels |
| `S` | Non-overlapping segment length, called `Ls` or `Lseg` in the paper |
| `L` | Number of observed segments; paper assumes `T/S` is an integer |
| `E` | Token/embedding width, unspecified numerically in paper |
| `h` | Attention head count; must divide `E` in conventional multihead attention |
| `R` | Learnable router count per segment, unspecified |
| `N` | Number of initial TDE layers in Algorithm 1, unspecified |
| `Le` | Number of encoder scales/decoder levels, unspecified |
| `H` | Number of future output steps; treating experimental `K` as `H` is an interpretation |
| `P` | Number of future decoder tokens, unspecified |
| `O` | Number of continuous output targets, unspecified |
| `C` | Number of stress classes, unspecified |

Recommended external contracts are input floating tensor `X[B,T,D]`, forecast `Yhat[B,H,O]`, and optional categorical logits with an explicitly chosen target convention. The output target must be supplied by the user/data protocol. Physiological measurements alone do not establish ground-truth stress labels.

Internally keep `[B,D,L,E]` so time and variable axes retain their separate semantics. A `[B,L,D,E]` layout is equally valid if permutations are correct. Never flatten `D` and `L` and call ordinary attention a faithful TDE implementation.

## 4. Module-by-module mathematical specification

### 4.1 MDSW: equations (1)-(3), page 5

For each variable independently, divide observations into consecutive, non-overlapping length-`S` blocks. Thus `[B,T,D] -> [B,D,L,S]`.

Apply a shared learnable projection `U: R^S -> R^E`, then add a fixed two-dimensional positional code identifying both segment index and variable identity:

`W[b,d,i,:] = U(Xsegment[b,d,i,:]) + position(i,d)`.

The result has shape `[B,D,L,E]`. The paper specifies fixed positional information but not its exact formula. A fixed sinusoidal encoding for the two axes is an engineering completion; learned encoder positions would not match the specific word "fixed". The decoder positions are separately described as learnable.

Unspecified: segment size, segment-projection bias, position-code formula, embedding initialization, extra embedding normalization/dropout, and handling of `T % S != 0`. The reference implementation should explicitly pad or reject incomplete segments rather than silently discard observations. Repeating the last observed value is a defensible documented padding convention and uses no future data.

### 4.2 TDE: equations (4)-(10), pages 5-6

Each TDE preserves `[B,D,L,E]`.

**Temporal stage.** Reshape to `[B*D,L,E]`, then share attention parameters over variables. Following the explicit residual/post-normalization equations:

`A = LayerNorm(F + TemporalAttention(F,F,F))`

`Ft = LayerNorm(A + TemporalFFN(A))`.

The attention operates only over the `L` observed segments of one variable. Standard head-wise dot-product attention uses `softmax(QK^T/sqrt(E/h))V`; the paper names MHSA without specifying a different kernel.

**Dimensional stage.** Rearrange to `[B*L,D,E]`. At each segment, learn `R` router tokens `delta[i,:,:]`, with shape `[L,R,E]` before broadcasting across the batch. Routers collect and then redistribute information:

`message = GatherAttention(Q=router, K=Ft, V=Ft)` with shape `[B*L,R,E]`.

`update = DistributeAttention(Q=Ft, K=message, V=message)` with shape `[B*L,D,E]`.

`A = LayerNorm(Ft + update)`.

`Fd = LayerNorm(A + DimensionalFFN(A))`.

Rearrange back to `[B,D,L,E]`. Equation (8) prints a self-referential `Fd` residual; using the temporal-stage features is the consistent sequential interpretation. The gather and distribute attentions are given distinct subscripts in equations (6)-(7); separate parameter sets are a natural implementation. The paper explicitly shares dimensional MHSA parameters across segment positions, not necessarily between these two operations.

Algorithm 1 repeats TDE `N` times before the hierarchy. Each TDE layer may have its own weights and router parameters. Exact layer sharing is not specified.

Unspecified: `R`, `E`, `h`, FFN hidden width/activation, dropout, attention masks, initializations, router initialization, and LayerNorm epsilon. Prose and Fig. 11 also show input normalization, while equations explicitly show postnorm residuals. Prefer the explicit equations and record the choice.

**Complexity correction.** Ignoring feature width, temporal attention costs `O(B*D*L^2)`. Router attention costs `O(B*L*D*R)`, or `O(B*L*D)` only for fixed `R`. Direct cross-variable attention would cost `O(B*L*D^2)`. The paper compares router cost with `O(D*L^2)`, which describes temporal rather than direct dimensional attention. Width projections and FFNs add separate costs, so this is not a full FLOP formula.

### 4.3 EDM encoder: equations (11)-(13), page 6

Algorithm 1 assigns the initial hierarchy feature directly from the initial TDE stack:

`Encoder[0] = TDE_stack(W)`, shape `[B,D,L0,E]`, `L0=L`.

For each deeper level, pair adjacent temporal segments and concatenate their embeddings along the feature axis:

`pair[i] = concat(Encoder[l-1][2i], Encoder[l-1][2i+1])`, shape `[B,D,Ll,2E]`.

Map `2E -> E` using a learnable linear map, then apply TDE at that scale. For even lengths `Ll = L(l-1)/2`. With explicitly documented tail replication, `Ll = ceil(L(l-1)/2)`.

The paper says concatenation is "along time" but its equation pairs two vectors and applies a learnable map. Flattening each two-token pair to `2E`, followed by projection to `E`, is the consistent implementation that actually halves the number of tokens.

**Indexing ambiguity:** equation (13) also applies TDE to the first encoder level, whereas Algorithm 1 directly assigns `F^N` to the first level. Use Algorithm 1's single initial stack rather than inadvertently adding an undocumented extra TDE. State this convention.

Unspecified: level count, whether widths change by scale, odd-length handling, merge normalization, number of TDE layers per deeper scale, and whether hierarchy can continue after length one.

### 4.4 EDM decoder: equations (14)-(18), page 6

Initialize a learnable future positional tensor `Jdec[B,D,P,E]` by broadcasting trainable parameters across the batch. This contains no true future observations or targets.

At each of `Le` decoder levels:

1. Run TDE on the current decoder feature, retaining `[B,D,P,E]`.
2. For each variable, use decoder tokens as queries and corresponding encoder-scale tokens as keys and values. Shapes are Q `[B*D,P,E]`, K/V `[B*D,Ll,E]`.
3. Apply residual LayerNorm and FFN with another residual LayerNorm.
4. Project this level's decoded feature to a prediction on a common output grid.

Sum the scale-wise predictions, as explicitly required by Algorithm 1: `Z = sum_l Yhat_l`.

Decoder depth and encoder scale count match in the algorithm. Equation indexing starts the decoder at zero but the encoder at one, and equations (14)-(16) interchange encoder/decoder superscripts. A clean zero-based implementation resolves these notation errors without changing the intended order.

The paper does **not** determine `P`, projection shape, target-channel count, whether decoder tokens represent samples or segments, or the temporal units of the output. One reasonable configurable completion is `P=ceil(H/S)`, per-token projection `E -> S`, reconstruction/cropping to `[B,H,D]`, and later channel projection to `O`. Another is `P=H` with one predicted step per token. These choices are not interchangeable claims about the original experiments; the package must document which it uses.

### 4.5 DGC: equation (19), page 6

The specified layer recurrence is:

`g_l = Transform(concat(g_0, g_1, ..., g_(l-1)))`.

The text specifies **two dense blocks** and says the transform includes a `1 x 3` convolution followed by `1 x q` max-pooling. DGC follows summed EDM predictions in Algorithm 1 and precedes the final prediction head.

No graph adjacency matrix, neighborhood construction, graph normalization, graph propagation equation, node assignment, or adjacency learning objective is supplied. Therefore ordinary densely connected temporal convolutions are the most literal executable interpretation of the actual operator definition. Calling an invented adjacency operator "the paper's DGC" would be unjustified.

Major shape gap: if pooling shrinks time at every dense layer, earlier feature maps cannot be concatenated with later maps without an alignment rule. The text supplies no such rule. Length-preserving pooling inside a dense block, or explicitly aligning every history tensor, is required for a well-defined implementation. A reference can use stride-one same-length pooling, state the padding convention, and preserve horizon length. This is an engineering resolution, not a documented author choice.

The same `q` is called both pooling extent and growth rate in the prose, but these are distinct concepts. Use separate configuration names for new channels per dense layer and pooling kernel size. Also document the number of layers per block, activations, convolution channels, transitions, and output projection. None are fully specified.

### 4.6 Prediction heads and losses

Algorithm 1 ends with `PredictionHead(DGC(Z))` to obtain a continuous stress sequence. The head architecture is unspecified. A linear map from final features to explicit target dimension `O` is a reasonable completion. Avoid an arbitrary bounded output activation unless the target scale warrants it.

The manuscript does not state a supervised loss formula. The current repository explicitly uses mean squared error, so MSE is a supported repository-informed regression objective:

`Lreg = mean((Yhat - Y)^2)`.

Classification metrics are reported, but the paper does not define class count, class boundaries, output head, sample versus sequence labels, classification loss, or regression-to-class conversion. An optional categorical head and cross-entropy objective are independent functionality, not a recovered paper design. Use explicit labels supplied with data; do not manufacture classes from the four driving contexts or blood pressure groups.

Fig. 9 names additional losses, but no formula, weight, teacher checkpoint, teacher update rule, pretraining data, or training schedule is provided. These losses should not be silently invented for the equation-based reference.

## 5. Data and preprocessing evidence

### Dataset A: self-collected driver signals, page 7

- Experiment section: 300 healthy drivers and more than 300,000 data points. Abstract/introduction instead say more than 300 taxi drivers. Exact subject count and study population need clarification.
- Four acquisition contexts: stationary vehicle with engine running, ordinary roads, viaducts, and tunnels. These are acquisition contexts, not specified stress classes.
- Named information: emotional pulse, respiration, environmental humidity signal (EHS), respiratory humidity signal (RHS), pressure pulse signal (PPS).
- Physical sensor description establishes three sensor channels, while later discussion names five input categories. The transformations from PPS to emotional pulse and RHS/EHS to respiration, and whether raw and derived variables are all retained, are unspecified.
- EHS is said to remove ambient interference from RHS, but no calibration, subtraction, regression, or filtering formula is given.
- Fig. 8 names synchronization, missing-data handling, filtering, windowing, feature engineering/selection, balancing, and train/test separation without numerical methods.

Missing: downloadable data, schema, consent/access terms, timestamps, sample rate, sensor units, per-subject/session durations, raw-to-feature preprocessing, artifact rejection, stress ground truth, annotator/questionnaire method, class boundaries, split memberships, split sizes, normalization, and sequence stride.

### Dataset B: public BCG used to derive HRV, page 7

Paper link: https://www.kaggle.com/datasets/bahadirarabaci/bcg-ds . The link was attempted during this audit, but the public page was not retrievable through the web tool; dataset contents were not independently inspected. Do not claim dataset downloading or a verified data schema.

The paper describes an RS-611 micromotion-sensitive mattress with two hydraulic pressure sensors, 16-bit conversion, and BCG sampled at 100 Hz. It states 175 university staff were recruited and 128 valid records remained, comprising 61 hypertensive and 67 normotensive participants. Table I instead lists 71 and 67, with the hypertensive sex counts also summing to 71. This is an unresolved cohort-size contradiction.

No BCG peak detector, beat-quality filter, RR/NN cleaning method, HRV window, HRV feature set, resampling/alignment scheme, or stress-label mapping is supplied. Hypertension is not an interchangeable stress ground truth. A loader that forecasts provided BCG/HRV values can be useful, but cannot be described as reproducing driver-stress classification without additional target evidence.

### Safe reference-pipeline requirements

These are engineering practices to avoid leakage, not recovered experiment settings:

1. Require explicit feature and target columns and document time units.
2. Preserve subject/session boundaries; never create windows that cross them.
3. Prefer explicit subject-disjoint splits when subject IDs are available. Otherwise use chronological splits and clearly disclose that subject-independent generalization is untested.
4. Split before fitting normalizers or creating overlapping windows. Fit normalization and imputation statistics on training data only.
5. Do not interpolate test information backward across split boundaries. Centered filters may leak future information within a forecasting setting; default to no filtering unless the acquisition protocol is known.
6. Forecast targets must follow the observation window. Do not supply future labels/observations as decoder inputs.
7. Persist preprocessing, feature order, label maps, target units, split IDs, and model configuration with checkpoints.
8. Clearly label synthetic demonstrations as plumbing tests; synthetic stress-like targets do not validate physiological stress prediction.

## 6. Training settings: known versus missing

| Setting | Evidence |
|---|---|
| Framework | PyTorch, paper page 7 |
| Accelerator | NVIDIA GeForce RTX 4060, page 7 |
| Optimizer | Adam, page 7 and repository config |
| Learning rate | 0.001, page 7 and repository config |
| Batch size | 32, page 7 and repository config |
| Lookback | 24, page 7; processed time unit unknown |
| Experimental K | 3, 6, 12, 24, pages 8-12; repeatedly called prediction setting, without a formal model-level definition |
| Regression loss | MSE in repository; not specified in paper |
| Embedding width | 128 in scaffold, absent from paper |
| Attention heads | 4 in scaffold, absent from paper |
| Input channel default | 5 in scaffold, actual raw/derived schema unresolved |
| Segment length | 24 in scaffold; absent from paper and degenerate with paper lookback 24 |
| Number of layers | Config says 3 but unused; exact interpretation unsupported |
| Forecast length | Config says 24 but unused; does not determine actual scaffold output length |

Unknown: epoch count, early stopping, validation metric, schedule, Adam betas/epsilon, weight decay, gradient clipping, precision, initialization, seed, repeated-run protocol, class balancing, target scaling, optimization of any auxiliary loss, all baseline hyperparameters, and model-selection procedure. Package defaults for these must be labeled as reference defaults.

## 7. Metrics and result audit

### Definitions, page 7

- `MAE = sum(abs(yhat-y))/n`.
- `MSE = sum((yhat-y)^2)/n`.
- The printed residual standard error is `RSE = sqrt(sum((y-yhat)^2)/(n-2))`, **not** the normalized forecasting RSE `sqrt(SSE/SST)`. It is undefined for `n <= 2`. Preserve its explicit name/formula if implementing it; do not clamp the denominator and silently change the statistic.
- `R2 = 1 - SSE/SST`, where `SST = sum((y-mean(y))^2)`. Constant-target handling and multioutput aggregation must be documented.
- Classification: accuracy, precision, recall, F1. Macro/micro/weighted/binary averaging and positive class are not supplied.
- Aggregation across horizon, variables, subjects, or windows is unspecified. A package should state its convention and preferably also report per-horizon/per-target values.

### Reported numerical anchors (not acceptance criteria)

- Classification: accuracy 95.1%, precision 92.3%, recall 95.8%, F1 94.1%. Dataset/split/class attribution is insufficient to reproduce these values.
- Dataset A full-model RSE for K 3/6/12/24: .2575/.2632/.3105/.3684; R2 .9849/.9789/.9682/.9495 (page 8 text).
- Dataset A MSE for the same K values: 2.668/2.846/2.879/2.891; MAE 1.145/1.124/1.074/1.095 (pages 10-11 text).
- Dataset B MSE: 2.594/2.759/2.731/2.686; MAE 1.169/1.175/1.074/1.178 (page 10 text).
- Table IV reports 2658 MB peak training GPU memory and MSE .3582 for PSDformer. It does not sufficiently specify the dataset, K, target scaling, memory-measurement method, or connection to the much larger MSE values above.
- Table V reports mean +/- sample standard deviation **across four K settings**, not repeated random seeds, folds, or uncertainty intervals. These statistics must not be relabeled as experiment-replication variance.

### Important contradictions and omissions

1. **Fig. 9 versus Algorithm 1 (page 4).** Fig. 9 is a two-part layout with pretrained weights, matching, feature regularization, fusion, and consistency loss. Algorithm 1 is a sequential MDSW -> TDE -> hierarchical EDM -> DGC -> head computation. No equations resolve the additional branches.
2. **Fig. 12 versus EDM text (page 6).** The image shows graph-node features, latent graph vectors, and an adjacency reconstruction output `[N,N]`. Equations/text describe temporal segment merging and future sequence decoding. Do not graft an adjacency reconstruction loss into EDM.
3. **DGC naming/operator ambiguity (page 6).** Heading repeats "Encoder-Decoder Module (DGC)"; transform is ordinary convolution/pooling; no graph definition. Growth and pooling both use q, and dense connections have no shape-alignment rule after downsampling.
4. **Normalization/index notation.** Algorithm 1 omits some residual details that the equations explicitly provide; decoder equations mix encoder and decoder superscripts; first-level TDE counts differ as discussed above.
5. **Fig. 17 versus caption/text/Table III (page 9).** Visually the first four panels use "Relative RSE (%)" and "R2 decrease (percentage points)" despite the caption/text referring to raw RSE and R2. The lower bars do not match Table III: for example panel (e) displays 5.62, 4.15, 7.58 for its three ablation labels, while Table III reports A-RSE penalties 12.42, 12.64, 18.48. Labels and ordering are also inconsistent. No transformation reconciles these in the text.
6. **Fig. 17 acronym expansions.** Caption calls MSED "multi-scale temporal embedding decoder", ED "efficient decomposition", and MS "multi-domain spatial weighting"; surrounding text says MS removes MDSW, ED removes EDM, and MSED removes both. Implement explicit module flags and regard exact author ablation wiring as unknown.
7. **Fig. 19 versus numerical discussion/Table V (page 11).** The red PSDformer Dataset A MSE trace is visually about 2.25 at K=3 and about 2.0 at K=12, whereas the text gives 2.668 and 2.879. Thus even the direction of the optimum differs. These plotted values are approximate readings; the exact text values are quoted above. Fig. 19's method ranks also differ from Table V's ordering.
8. **Incompatible MSE/MAE pairs (Table V).** For the same errors and units, `MAE <= sqrt(MSE)`. Dataset A GRU is reported as MSE 6.808, MAE 4.641, and RL-LSTM as MSE 7.596, MAE 5.513; these violate the inequality. Averaging over K does not remove the incompatibility when the aggregation/units match. Different undocumented scales or mislabeled values would need clarification.
9. **Cohort inconsistency.** Dataset B text totals 128 but Table I totals 138; Dataset A says 300 in experiments versus more than 300 elsewhere.
10. **Undefined targets and task alignment.** Continuous stress labels, class labels, source of ground truth, temporal alignment, target scale, and labels for the BCG cohort are not specified.
11. **Unspecified attribution/feature plots.** Fig. 17 labels channel contribution as attention weights but gives no aggregation or attribution algorithm; representations and illustrative stress plots cannot be regenerated from a specified procedure.
12. **Simulation is not external validation.** Fig. 20 is explicitly a simulated driving/stress scenario. It cannot substitute for held-out real-driver performance.

Consequently, reported performance numbers should be retained as claims from the manuscript, not executable acceptance thresholds. Correctness checks should verify shapes, equations, gradients, leakage protections, checkpoint round trips, and genuine held-out evaluation on user-provided data.

## 8. Minimum reference implementation and test checklist

### Required functionality

- Independent package with clean imports and documented installation/CLI.
- MDSW with non-overlapping per-variable segments and two-axis positional encoding.
- Temporal attention plus learnable router gather/distribute TDE with residual FFNs.
- Hierarchical pair-merging encoder and learned-query multiscale decoder, with scale predictions summed before DGC.
- Two dense convolution blocks and an explicit pooling/shape policy.
- Configurable continuous forecast head; optional classification labeled as an extension.
- Explicit CSV/array data schema, leak-resistant splits/windows/scalers, real training/validation/test loops, selected best-validation checkpoint, inference, and metrics.
- Future extension for study-level replication: separate no-MDSW, no-EDM, and joint-ablation configurations. These are not included in this package because the manuscript does not specify the replacement operators; the delivered model implements the full equation-based pipeline only.
- Synthetic smoke demonstration requiring no private dataset or pretrained weights.

### Verification cases

- Shape round trips for D=1 and D>1; batch size one; nondivisible lookback/horizon; odd segment counts; horizon one.
- Attention weight/projection dimensions; E divisible by h; routers never mix examples; no target data in decoder inputs.
- No flattening of dimension/time axes before the two TDE stages; shared attention weights along the intended axes.
- Encoder lengths follow the documented merge/pad recurrence; decoder cross-attention uses matching variables and the intended encoder scale.
- Dense block concatenations remain shape compatible, and final output is exactly `[B,H,O]`.
- Backpropagation reaches MDSW, temporal attention, routers, dimensional attention, every encoder/decoder scale, DGC, and output head.
- Finite loss/gradients and small-batch overfitting on a deterministic synthetic task; no claim of meaningful biomedical accuracy from this test.
- Scalar metric hand-checks including RSE's `n-2` denominator and undefined edge cases, constant targets, and classification averaging.
- Scaler fitted only on training data; split/window boundaries checked; no windows crossing subject/session boundaries.
- Save/load predictions agree in eval mode, configurations and preprocessing persist, and CLI train/evaluate/predict work from a fresh package import.

## 9. Information needed for an exact experimental reproduction

Request the authors' actual experiment code/commit and full configs; Dataset A and its access conditions; Dataset B record IDs and preprocessing code; exact participant counts; label-generation/annotation protocol; all training/validation/test memberships; sample rates and processed time-step units; class count and averaging rules; MDSW segment/position settings; TDE router/head/FFN/depth settings; exact encoder/decoder scale and projection shapes; DGC graph or convolution definition and pooling alignment; final head; all loss equations/weights; epochs/checkpoint selection; and corrected figures/tables. Until those exist, the defensible deliverable is a transparent, tested reference implementation of the described equation-based architecture.

## Appendix: concrete independent model choices

The accompanying reference model was inspected after implementation. The following choices resolve gaps above; they are **not claimed paper settings**. Training, smoke, and other presets may override them, so a saved configuration is the authoritative description of an individual run.

- Model dataclass defaults: `D=5`, `T=24`, `H=3`, `O=1`, `E=64`, four attention heads, segment length `S=4`, four routers per segment, one initial TDE layer, three encoder/decoder levels, dropout 0.1, and FFN width `4E` with GELU. The optional demonstration classifier has three outputs and may be disabled; three classes are not established by the paper.
- MDSW concatenates fixed sinusoidal variable-axis and segment-axis positional vectors, splitting the embedding width between them. Incomplete input segments are **zero padded**. Odd pair-merging lengths repeat the final token. These are two distinct boundary policies.
- With defaults, the encoder token counts are `6 -> 3 -> 2`. The first memory is the initial TDE output, following Algorithm 1. Deeper memories apply merge then TDE.
- The learned decoder position tensor is `[1,D,ceil(H/S),E]`, initialized normally with standard deviation .02 and broadcast across batches. Decoder levels consume encoder memories fine-to-coarse. Each projects `E -> S`, reshapes to a future sequence, crops to `H`, and contributes to the sum `[B,D,H]`.
- DGC views that sequence as `[B,1,D,H]`, with variables on the spatial-height axis. Each of two dense blocks has two layers, growth 16, temporal `1 x 3` Conv2d, and length-preserving stride-one max-pooling of width two inside each layer. One additional temporal max-pooling operation with stride two and ceiling mode at each block exit compresses the time axis **between** blocks. This explicit alignment scheme is necessary engineering detail not specified in the manuscript.
- Dense channel counts are `1 -> 33 -> 65` across the two blocks. For default `H=3`, compressed time lengths are `3 -> 2 -> 1`, producing `[B,65,D,1]` before the task head.
- A flattened DGC feature is projected to `E`, followed by GELU/dropout. Forecast projection maps `E -> H*O`, then reshapes to `[B,H,O]`. Thus the final forecast is a learned projection of the compressed multiscale representation, rather than simply an identity correction to the pre-DGC sequence. The optional classifier returns one label distribution per input window, `[B,C]`; it is not per-future-step classification.
- Diagnostics expose pooled features `[B,E]`, summed pre-DGC predictions `[B,H,D]`, and the individual scale predictions `[B,Le,H,D]`. The latter allow checking that the sum is exact.
- Missing or nonfinite input values are replaced with zero at the model boundary. The optional boolean observation mask marks available samples; it is not an attention mask or a learned missingness embedding. Preprocessing and any imputation/scaling belong outside the model and must use training-only statistics.
- There is no graph adjacency, pretrained teacher, cross-modal matching loss, feature-regularization loss, or output-consistency loss. These omissions are intentional and correspond to the unresolved evidence gaps above.

This appendix records architecture choices, not a declaration that every test in the checklist has run. Actual verification results should be reported by the package's tests and execution report.
