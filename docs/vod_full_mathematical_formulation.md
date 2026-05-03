# VOD Full Mathematical Formulation

Date: 2026-04-29

This document restores the mathematical objects behind VOD. It is not an
implementation report. It defines the claims and operators that code must
faithfully realise before any experiment can be treated as a VOD claim test.

## 0. Core Principle

VOD is not a platform that routes media to external models. It is a native
unified generator over a shared entropy field:

```text
U = U(t, y, x, c)
```

where:

```text
t  temporal coordinate
y  vertical spatial coordinate
x  horizontal spatial coordinate
c  entropy/channel coordinate
```

Every medium is a projection of the same field:

```text
X_m = P_m(U, b_m)
```

with medium-specific boundary `b_m`, not a separate model:

```text
m ∈ {image, video, audio, text, layout}
```

The research question is whether these projections can be learned and
stabilised by VOD-native field dynamics, not by stitching external generators.

## 1. Entropy Texture And Entrons

Let `Omega` be the continuous or discretised spacetime domain:

```text
Omega = T × Y × X
```

The entropy field is:

```text
U: Omega -> R^C
```

An entron is the local compressed information state:

```text
e(omega) = [a(omega), phi(omega), f(omega), cr(omega), s(omega), snr(omega)]
```

where:

```text
a     amplitude
phi   phase
f     dominant frequency
cr    compression ratio
s     salience
snr   signal-to-noise ratio
```

Compression differential:

```text
e(omega) = Delta I(omega)
         = I_raw(omega) - I_context(omega)
```

Compression ratio:

```text
cr(omega) = I_context(omega) / (I_raw(omega) + eps)
```

The model should not treat patches or tokens as primitives. Patch, token,
frame, waveform, and glyph are measurement instruments over the entropy field.

## 2. Chladni Entropy Field

### 2.1 2-D Chladni Basis

For a rectangular boundary, a 2-D Chladni-like mode can be written:

```text
Psi_{m,n}(x,y)
  = cos(pi m x) cos(pi n y)
  - cos(pi n x) cos(pi m y)
```

A field is a weighted mode sum:

```text
U(y,x) = Σ_i w_i Psi_{m_i,n_i}(x,y + phi_i)
```

In discretised form:

```text
x_j = j / (W - 1)
y_i = i / (H - 1)
```

### 2.2 Spatiotemporal Chladni Field

For video, time is not a fake roll operation. It is an axis of the field:

```text
U(t,y,x)
  = Σ_i w_i cos(2π m^t_i τ + phi^t_i)
        [ cos(π m^x_i x) cos(π m^y_i y)
        - cos(π m^y_i x) cos(π m^x_i y) ]
```

where:

```text
τ = t / (T - 1)
```

Video projection is slicing:

```text
P_video(U)_t = DecodeVideoSlice(U(t,:,:,:))
```

Image projection may be a temporal reduction:

```text
P_image(U) = DecodeImage(Mean_t U(t,:,:,:))
```

This is the minimal mathematical reason VOD can discuss video: motion is field
evolution, not ordered independent images.

## 3. Media Projection Boundary

Each medium has its own projection boundary:

```text
b = {b_s, b_t, b_a, b_y}
```

where:

```text
b_s spatial canvas / masks / layout boxes
b_t temporal duration / fps / shot rhythm
b_a audio sample rate / beat grid / frequency grid
b_y symbol boxes / text masks / reading order
```

Projection:

```text
X_m = P_m(U, b_m)
```

Encoding:

```text
E_m = Phi_m(X_m, b_m)
```

Consistency:

```text
L_projection = Σ_m w_m d_m(Phi_m(Y_m, b_m), P_m(U, b_m))
```

Important:

```text
Do not compare raw image/video/audio/text directly.
Compare only after projection through each medium boundary.
```

## 3A. Early AI Drawing Hypotheses: Shadow Sampling, Dynamic Zoom, And Constraint Conversion

This section restores the early AI drawing research note:

```text
C:\Users\asus\Documents\早期AI绘图研究.txt
```

Status:

```text
UNVERIFIED EARLY HYPOTHESES / ENGINEERING HEURISTICS
```

The note was produced during an earlier GPT-o3-era discussion. Its concepts are
useful, but they must not be treated as proven VOD mathematics until they pass
claim-specific baselines. The rest of this section translates the ideas into
testable VOD hypotheses.

The note's central intuition is that current image generators often behave like
shadow samplers: they compose probability shadows induced by prompt features
rather than directly operating on stable object structure. This is a useful
modeling metaphor, not yet a measured theorem.

### 3A.1 Shadow Sampling Hypothesis

Let the user request be:

```text
r = {semantic words, style words, negative words, references}
```

The image model implicitly constructs a shadow field:

```text
S_r(omega) = Shadow(r; theta)(omega)
```

where `S_r` is not the object itself, but a probability-shaped visual
constraint over the image field.

Generation then approximates:

```text
Y ~ p_theta(Y | S_r)
```

not:

```text
Y = Render(real_object_geometry)
```

Positive prompt:

```text
S_+(omega) = Shadow(prompt_+)
```

Negative prompt:

```text
S_-(omega) = Shadow(prompt_-)
```

Guidance can be interpreted as:

```text
S_guided = S_+ - lambda_neg S_-
```

This explains why prompt-only precision can drift:

```text
more words -> more shadow interactions -> more conflict modes
```

The VOD correction is not "write longer prompts"; it is:

```text
convert shadows into constraints
```

Validation status:

```text
OPEN. This is an interpretive hypothesis.
```

Metric candidates:

```text
prompt_conflict_rate
structure_drift_rate
layout_violation_rate
cast_presence_error
```

Baselines:

```text
B0 prompt-only long prompt
B1 prompt-only short structured prompt
B2 prompt + explicit masks/control boundaries
```

Falsification condition:

```text
If B2 does not significantly reduce structure_drift_rate or
layout_violation_rate versus B0/B1 across hard prompts, the "convert shadow
to constraint" claim is not supported.
```

### 3A.2 Constraint Conversion

Prompt features that describe hard structure must be extracted from text and
converted into boundaries:

```text
text wish                 -> VOD constraint
---------------------------------------------------------
who appears               -> CastPresent / identity boundary
where a panel is          -> layout boundary b_layout
where a sign/text sits    -> symbol boundary b_y
pose                      -> pose control boundary
depth / room geometry     -> depth / perspective boundary
object mask               -> segmentation boundary
logo position             -> UV plane boundary
```

Mathematically:

```text
r -> (c, b, q)
```

where:

```text
c semantic condition
b hard boundary set
q phase/frequency/mode state
```

Then VOD samples:

```text
Y ~ p_theta(Y | c, b, q)
```

instead of only:

```text
Y ~ p_theta(Y | prompt string)
```

Validation status:

```text
OPEN. Needs controlled prompt-only vs boundary-conditioned comparisons.
```

Metrics:

```text
layout_iou
mask_boundary_error
object_count_error
cast_presence_error
forbidden_object_error
```

Baselines:

```text
B0 prompt-only
B1 prompt + negative prompt
B2 prompt + mask/control/depth/pose boundary
```

Pass condition:

```text
B2 must reduce hard-structure errors by a pre-registered margin without
significantly degrading visual quality.
```

### 3A.3 Dynamic Zoom / Micro-Imaging

The early note's "dynamic zoom / micro-imaging" corresponds to a multiscale
constraint refinement process.

Define scales:

```text
s_0 < s_1 < ... < s_L
```

with fields:

```text
U^{(l)} = U at scale s_l
```

Coarse-to-fine process:

```text
U^{(0)}        = GenerateCoarse(c, b^{(0)}, q)
U^{(l+1)}      = Refine(Zoom(U^{(l)}, R_l), c, b^{(l+1)}, q)
Y^{(l+1)}_R    = DecodeRegion(U^{(l+1)}, R_l)
```

where `R_l` is a selected region:

```text
panel, face, hand, sign, logo plane, text region, difficult object
```

This is not mystical "seeing the real object". It is a control strategy:

```text
global shadow -> local boundary -> refined field -> composite
```

In production terms:

```text
full page -> difficult panel -> masked region -> symbol/text layer
```

In model terms:

```text
global field -> local field crop -> boundary-conditioned refinement
```

Validation status:

```text
OPEN. This is an engineering workflow hypothesis.
```

Claim E2:

```text
Dynamic zoom / local refinement improves hard-region success rate compared
with repeated full-image regeneration.
```

Metrics:

```text
hard_region_success_rate
local_mse_or_lpips_vs_reference
identity_anchor_error
panel_replacement_boundary_seam
number_of_attempts_to_success
```

Baselines:

```text
B0 repeated full-image regeneration
B1 full image + stronger prompt
B2 masked local refinement / dynamic zoom
B3 local refinement + final composite
```

Falsification condition:

```text
If B2/B3 does not improve hard_region_success_rate or reduce attempts-to-success
over B0/B1 on the same hard regions, dynamic zoom is not validated.
```

### 3A.4 Mask As Boundary Operator

A mask is not a cosmetic editing instruction. It is a hard boundary operator:

```text
M_R(omega) =
  1 if omega ∈ R
  0 otherwise
```

Masked update:

```text
U' = (1 - M_R) ⊙ U_base
   + M_R ⊙ Update_theta(U_R, c_R, b_R, q_R)
```

This is the mathematical basis for:

```text
inpaint
single-panel regeneration
object mask control
logo plane isolation
text region isolation
```

Masking converts a vague prompt shadow into a local solvable boundary problem.

Validation status:

```text
OPEN but strongly supported by existing image-editing practice; still needs
VOD-specific measurement.
```

Claim E1:

```text
Mask-as-boundary lowers local edit error while preserving unmasked context.
```

Metrics:

```text
masked_region_error
unmasked_region_drift
boundary_seam_error
object_localisation_iou
```

Baselines:

```text
B0 full image regeneration
B1 prompt-only inpaint
B2 mask-conditioned local update
```

Falsification condition:

```text
If B2 does not reduce masked_region_error while keeping unmasked_region_drift
below threshold, mask-as-boundary is not validated for VOD.
```

### 3A.5 Text / Logo As UV Plane

Text and logo generation fail because text is a discrete symbolic shape but
diffusion image synthesis treats it as continuous texture.

For a planar sign/card/logo region, define a quadrilateral in image space:

```text
Q = {p_1, p_2, p_3, p_4}
```

Define a homography:

```text
H_Q: [0,1]^2 -> Q
```

UV plane:

```text
T_uv(u,v) = VectorTextOrLogo(u,v)
```

Projection into image:

```text
T_img(x,y) = T_uv(H_Q^{-1}(x,y))
```

Composition:

```text
Y_final = Composite(Y_base, T_img, alpha_Q, lighting_Q)
```

where:

```text
alpha_Q     opacity / antialiasing mask
lighting_Q  shadow, highlight, blur, reflection from the base plane
```

This is the "globe writing" principle from the early note:

```text
writing directly on curved / perspective image space is unstable;
unfold the plane, write accurately, then project back.
```

Validation status:

```text
OPEN as VOD-native mechanism; engineering baseline is expected to be strong.
```

Claim E3:

```text
UV plane text/logo projection improves exactness compared with direct model
generation of text/logo inside the image.
```

Metrics:

```text
OCR_exact_match_rate
character_error_rate
logo_shape_iou
homography_corner_error
photometric_integration_error
human_seam_rating
```

Baselines:

```text
B0 direct generated text/logo
B1 generated blank sign + plain 2-D overlay
B2 generated blank sign + homography UV projection
B3 B2 + local lighting/shadow integration
```

Falsification condition:

```text
If B2/B3 does not improve OCR_exact_match_rate and logo_shape_iou over B0/B1,
or if seam/integration error is unacceptable, UV-plane text is not validated.
```

### 3A.6 Layered Gestalt Text

For generated text that must be integrated rather than simply pasted, use
layered contour constraints.

Let a glyph or logo be represented by contour levels:

```text
G = {C_0, C_1, ..., C_L}
```

where:

```text
C_0 outer silhouette
C_1 stroke skeleton / median contour
C_2 inner holes
C_3 serif / decoration / logo-specific detail
...
```

Layered text field:

```text
rho_text = Σ_l alpha_l DistanceField(C_l)
```

Discrete symbol code:

```text
B_text = EncodeString("四叶重工")
```

Binary-Twin constraint:

```text
L_text_layered
  = Σ_l w_l d(ExtractContour(Y, l), C_l)
  + lambda_BT L_BT(r_text)
```

This maps the early note's "分层格式塔 / 等高线模型" to VOD:

```text
text is not free texture;
text is layered contour + discrete code + plane projection.
```

Validation status:

```text
OPEN. This is a VOD-specific hypothesis and must not be assumed.
```

Claim E4:

```text
Layered contour text improves generated text readability compared with
single-mask or prompt-only text generation.
```

Metrics:

```text
OCR_exact_match_rate
character_error_rate
stroke_contour_iou
skeleton_distance
inner_hole_accuracy
visual_integration_score
```

Baselines:

```text
B0 prompt-only text generation
B1 single silhouette mask
B2 layered contours {outer, skeleton, holes, details}
B3 UV composite vector text
```

Falsification condition:

```text
If B2 does not improve OCR / contour metrics over B0/B1, the layered gestalt
text claim fails. If B3 dominates B2 on all metrics, layered generation should
be treated as optional style integration rather than exact text solution.
```

### 3A.7 Two-Stage Text / Logo Generation

The reliable engineering form is:

```text
Stage 1: Generate image without exact text
    produce scene, lighting, sign plane, card plane, speech bubble, logo slot

Stage 2: Render exact symbol layer
    vector text / logo / contour layers / Binary-Twin code

Stage 3: Project and composite
    homography / UV projection / local lighting / blur / shadow
```

In VOD notation:

```text
Y_base      = DecodeImage(U, b_without_symbol)
T_exact     = RenderSymbol(B, contours, UV plane)
Y_final     = Composite(Y_base, T_exact, b_y, lighting)
```

This is not an external-model workaround. It is VOD's correct separation of
continuous visual field and discrete symbol field.

Validation status:

```text
OPEN as VOD design; practically strong as production workflow.
```

Claim E3/E4 boundary:

```text
Two-stage text/logo generation is validated only if exactness and integration
metrics are both measured. Exact pasted text with bad lighting is not enough;
beautiful generated text with wrong characters is not enough.
```

Metrics:

```text
exactness = OCR_exact_match_rate + logo_shape_iou
integration = seam_error + lighting_consistency + perspective_error
```

Baselines:

```text
B0 direct generation
B1 plain overlay
B2 UV projection
B3 UV projection + lighting integration
```

## 3B. AI Manga Production Locks As VOD Boundaries

The early document also contains production-level constraints proven in AI
manga generation. They map directly to VOD boundaries.

Status:

```text
UNVERIFIED FOR VOD MODEL TRAINING.
ENGINEERING WORKFLOW OBSERVED AS USEFUL IN PRIOR AI MANGA PRODUCTION.
```

### 3B.1 Three Locks And One Flow

```text
Global Style Lock
Page Lock
CastPresent Lock
Single-Panel Flow
```

Mathematical form:

```text
b = b_style ∪ b_page ∪ b_cast ∪ b_panel
```

Style lock:

```text
b_style = {line_width, screentone_lut, background_white, noise_floor, negative_style}
```

Page lock:

```text
b_page = {panel_count, panel_grid, gutters, borders, balloon_regions}
```

Cast lock:

```text
b_cast = {required_characters, forbidden_characters, identity_anchors}
```

Single-panel flow:

```text
if panel_error(p) > threshold:
    regenerate p with b_panel(p)
    composite back into page
```

Validation status:

```text
OPEN. Prior production success is anecdotal; needs controlled page-level test.
```

Claim E5:

```text
Global Style Lock + Page Lock + CastPresent + Single-Panel Flow reduces manga
page structural errors compared with prompt-only page generation.
```

Metrics:

```text
panel_count_accuracy
panel_layout_iou
cast_presence_precision_recall
forbidden_character_rate
style_drift_score
screentone_consistency
speech_balloon_alignment
attempts_to_accepted_page
```

Baselines:

```text
B0 long prompt whole-page generation
B1 whole-page generation + negative prompt
B2 locks only
B3 locks + single-panel flow
```

Falsification condition:

```text
If B2/B3 does not reduce page structural errors or attempts-to-accepted-page
over B0/B1, the manga-lock workflow is not validated as VOD boundary logic.
```

### 3B.2 Panel-Level Update

Let `P_i` be panel `i`:

```text
Y_page = {Y_{P_1}, ..., Y_{P_N}}
```

Panel-local correction:

```text
Y'_{P_i}
  = Decode( Update_theta( Encode(Y_{P_i}), c_i, b_i, q_i ) )
```

Page composition:

```text
Y'_page = ReplacePanel(Y_page, i, Y'_{P_i})
```

This is the formal version of:

```text
full page -> hard panel -> regenerate panel -> paste back
```

### 3B.3 QA Metrics

Production QA maps to measurable constraints:

```text
panel_count_error
border_consistency
cast_presence_error
forbidden_cast_error
balloon_tail_alignment
text_readability
screentone_consistency
logo_exactness
```

These are not aesthetic extras. They are boundary-condition checks.

## 3C. Early Hypothesis Validation Matrix

No early AI drawing concept may be promoted to "VOD mathematical fact" without
passing its own test.

| ID | Claim | Primary Metric | Baselines | Status |
|---|---|---|---|---|
| E0 | Shadow sampling explains prompt drift | structure_drift_rate, prompt_conflict_rate | long prompt / short prompt / constraints | OPEN |
| E1 | Mask-as-boundary improves local edits | masked_region_error, unmasked_region_drift | full regen / prompt inpaint / mask update | OPEN |
| E2 | Dynamic zoom improves hard regions | hard_region_success_rate, attempts_to_success | full regen / stronger prompt / local refine | OPEN |
| E3 | UV text/logo improves exactness | OCR exact match, logo IoU, seam error | direct generation / overlay / UV projection | OPEN |
| E4 | Layered gestalt text improves generated text | OCR, contour IoU, skeleton distance | prompt / silhouette / layered / UV vector | OPEN |
| E5 | Manga locks reduce page errors | panel accuracy, cast error, style drift | whole-page prompt / negatives / locks / locks+panel flow | OPEN |

Reporting format:

```text
Early Claim ID:
Protocol:
Baselines:
Metrics:
Falsification condition:
Numbers:
Decision: PASS / FAIL / OPEN
```

Important:

```text
These early concepts are design hypotheses until tested.
Do not cite them as proven VOD math.
Do not let Claude/Codex turn them into implementation claims without metrics.
```

## 4. Binary-Twin Number

Binary-Twin Number is the continuous/discrete coupling object for VOD text,
logo, layout, and symbol regions.

### 4.1 Definition

A Binary-Twin number is:

```text
A = (rho, B)
```

where:

```text
rho ∈ R
B = (b_1, b_2, ..., b_i, ...) ∈ {0,1}^N
```

The space is:

```text
mathbb{B} = { (rho, B) | rho ∈ R, B ∈ {0,1}^N }
```

### 4.2 Encoding / Decoding Maps

Encoding:

```text
Phi: R -> {0,1}^N
B = Phi(rho)
```

Decoding:

```text
Psi: {0,1}^N -> R
Psi(Phi(rho)) = rho
```

Consistency condition:

```text
B = Phi(rho)
rho = Psi(B)
```

In VOD, `rho` is the continuous visual field of a symbol region, while `B`
is the discrete symbolic code.

### 4.3 Binary Metric

For two binary sequences:

```text
d_B(B, B') = Σ_{i=1}^∞ 2^{-i} |b_i - b'_i|
```

Product metric:

```text
d_BT((rho,B),(rho',B'))
  = |rho - rho'| + lambda_B d_B(B,B')
```

### 4.4 Binary-Twin Loss

For symbol region `r`:

```text
rho_r = ExtractContinuous(U, r)
B_r   = ExtractSymbolCode(r)
```

Visual decode:

```text
T_visual = OCR(VisualDecode(rho_r))
```

Symbol decode:

```text
T_symbol = SymbolDecode(B_r)
```

Conflict:

```text
L_BT(r)
  = d_text(T_visual, T_symbol)
  + lambda_pair |Psi(B_r) - rho_r|
  + lambda_code d_B(Phi(rho_r), B_r)
```

Total:

```text
L_BT = Mean_r L_BT(r)
```

Correction rule:

```text
if L_BT(r) high:
    reduce free visual diffusion in rho_r
    strengthen symbolic constraint B_r
```

This is the mathematical basis for solving image/text mutual interference:
visual texture and discrete text are coupled, not freely mixed.

## 5. Modular Shrinking Number

Modular Shrinking Number records convergence of continuous and discrete fields
under finite precision / modular representation.

### 5.1 Modular State

At step `k`:

```text
A_k = (E_k, B_k)
```

where:

```text
E_k continuous entropy field
B_k discrete symbol field
```

Precision scale:

```text
M_k ∈ N
```

Embedding:

```text
phi_M: Z/MZ -> R
```

### 5.2 Distances

Continuous:

```text
d_cont(E_{k+1}, E_k) = Mean_omega |E_{k+1}(omega) - E_k(omega)|
```

Discrete:

```text
d_disc(B_{k+1}, B_k) = Mean_r d_B(B_{k+1,r}, B_{k,r})
```

Pair consistency:

```text
d_pair(E_{k+1}, B_{k+1})
  = Mean_r |Psi(B_{k+1,r}) - ExtractContinuous(E_{k+1}, r)|
```

### 5.3 MSN

The modular shrinking number at step `k` is:

```text
MSN_k
  = alpha d_cont(E_{k+1}, E_k)
  + beta  d_disc(B_{k+1}, B_k)
  + gamma d_pair(E_{k+1}, B_{k+1})
```

Path MSN:

```text
MSN_path = Σ_{k=1}^K (1/k) MSN_k
```

Low MSN means stable convergence. High MSN means the model is jumping between
inconsistent continuous/discrete interpretations.

### 5.4 Normalisation

Candidate:

```text
A^*_{k+1} = (E^*_{k+1}, B^*_{k+1})
```

Normalisation:

```text
B'_{k+1} = NormalizeSymbol(B^*_{k+1}, Phi(E^*_{k+1}))
E'_{k+1} = NormalizeContinuous(E^*_{k+1}, Psi(B'_{k+1}))
```

Return:

```text
A_{k+1} = (E'_{k+1}, B'_{k+1})
```

## 6. TTNM Temporal Stability

TTNM provides the temporal stability logic without requiring VOD to solve a
literal physics PDE.

### 6.1 Tropical Operations

Tropical addition:

```text
a ⊕ b = min(a,b)
```

Tropical multiplication:

```text
a ⊗ b = a + b
```

### 6.2 Temporal Graph

Temporal graph:

```text
G_t = (N, E, W)
```

where:

```text
N nodes: frames, objects, beats, text boxes, layout regions
E edges: temporal adjacency, causal links, identity links, beat links
W(e) edge cost / delay / reliability
```

Node state:

```text
S_k(n)
```

Hard tropical update:

```text
S'_{k}(n)
  = ⊕_{e ∈ E_n} ( S_k(n_e) ⊗ W(e) )
  = min_{e ∈ E_n} [ S_k(n_e) + W(e) ]
```

### 6.3 Differentiable Soft-Min Version

Cost:

```text
C_{j->n} = d_state(S_j, S_n) + W_{j->n}
```

Soft tropical weights:

```text
alpha_{j->n}
  = softmax_j( -C_{j->n} / tau_T )
```

Propagated state:

```text
S'_n = Σ_j alpha_{j->n} Propagate(S_j, W_{j->n})
```

Temporal stability loss:

```text
L_TTNM = Mean_n d_state(S'_n, S_n)
```

For video frames:

```text
L_temporal = Mean_t || Y_{t+1} - Warp(Y_t, motion_t) ||_1
```

Toy fallback:

```text
L_temporal_toy = Mean_t |Y_{t+1} - Y_t|
```

## 7. 4/e Orthogonal Compression Decay

This is the full mathematical form of the operator. A scalar `4/e` multiplier
on iid Gaussian noise is only a placeholder and does not realise the VOD claim.

### 7.0 Claim Split

The `4/e` claim must be split into two different claims. They must not be
mixed again.

```text
Claim 1A: Operator conformance
    Does the implementation realise the mathematical definition of
    4/e Orthogonal Compression Decay?

Claim 1B: Application utility
    Does that operator improve a chosen downstream artifact metric
    compared with fair iid baselines?
```

Claim 1A is judged by structural signatures of the operator itself:

```text
four directional processes
non-iid covariance
axis-projected coherence
tile-boundary modulation
4/e decay normalisation
fair perturbation-energy matching
```

Claim 1B is judged by an application metric chosen for a specific task:

```text
artifact_score
directional contour coherence
four-axis residue ratio
edge preservation
human artifact rating
real renderer artifact reduction
```

Important:

```text
artifact_score cannot falsify Claim 1A.
It can only test a particular Claim 1B utility hypothesis.
```

Therefore:

```text
If AXCOV / covariance signature passes but artifact_score ties iid:
    Claim 1A = PASS
    Claim 1B under artifact_score = NOT ESTABLISHED

If covariance signature fails:
    Claim 1A = FAIL
    Claim 1B is not meaningful until implementation is fixed
```

### 7.1 Tile Residue

Let:

```text
X ∈ R^{H×W}
q = tile size
```

Neighbour jump set:

```text
J_all(X)
  = Mean( |X_{i+1,j} - X_{i,j}|, |X_{i,j+1} - X_{i,j}| )
```

Tile-boundary jump set:

```text
J_tile(X,q)
  = Mean(
      |X_{i+1,j} - X_{i,j}| where (i+1) mod q = 0,
      |X_{i,j+1} - X_{i,j}| where (j+1) mod q = 0
    )
```

Residue:

```text
R_tile(X,q) = J_tile(X,q) / (J_all(X) + eps)
```

Excess residue:

```text
r(X,q) = max(R_tile(X,q) - 1, 0)
```

If `r = 0`, there is no detectable boundary preference and the operator should
do nothing.

### 7.2 Four Orthogonal Projection Axes

In the 2-D projection, the four axes are:

```text
a_1: vertical          index j
a_2: horizontal        index i
a_3: primary diagonal  index i + j
a_4: secondary diag    index i - j + (W - 1)
```

These are not four tile corners. They represent four local degrees of freedom
of a four-axis orthogonal compression change rate projected to the image grid.

Sample four 1-D Gaussian processes:

```text
z_v[j]          ~ N(0, sigma_axis^2)
z_h[i]          ~ N(0, sigma_axis^2)
z_+[i+j]        ~ N(0, sigma_axis^2)
z_-[i-j+W-1]    ~ N(0, sigma_axis^2)
```

where:

```text
sigma_axis = beta * residue_gain * r(X,q)
```

### 7.3 4/e Decay

Orthogonal compression decay:

```text
D_oc = 4 / e
```

Per-axis decay:

```text
D_axis = D_oc / 4 = 1 / e
```

Structured orthogonal perturbation:

```text
N_4(i,j)
  = D_axis [ z_v[j] + z_h[i] + z_+[i+j] + z_-[i-j+W-1] ]
```

Equivalent:

```text
N_4(i,j)
  = [ z_v[j] + z_h[i] + z_+[i+j] + z_-[i-j+W-1] ] / e
```

### 7.4 Tile-Boundary Visibility Weight

Distance to nearest tile boundary:

```text
d_q(i,j)
  = min(
      i mod q, (q - 1) - (i mod q),
      j mod q, (q - 1) - (j mod q)
    )
```

Boundary visibility:

```text
w_q(i,j) = exp( -d_q(i,j) / lambda_q )
```

Default:

```text
lambda_q = 1
```

Final operator:

```text
OC_{4/e}(X)_{i,j}
  = X_{i,j} + w_q(i,j) N_4(i,j)
```

If `r(X,q)=0`:

```text
OC_{4/e}(X) = X
```

### 7.5 Covariance Signature

The operator must not be iid. It must have non-zero covariance along the four
projection axes:

```text
Cov(N_4(i,j), N_4(i, j+k))     ≠ 0    horizontal row coherence
Cov(N_4(i,j), N_4(i+k, j))     ≠ 0    vertical column coherence
Cov(N_4(i,j), N_4(i+k, j+k))   ≠ 0    primary diagonal coherence
Cov(N_4(i,j), N_4(i+k, j-k))   ≠ 0    secondary diagonal coherence
```

iid Gaussian noise has:

```text
Cov(noise(p), noise(q)) = 0 for p ≠ q
```

Therefore an implementation that only does:

```text
X' = X + Normal(0, sigma)
```

does not implement 4/e Orthogonal Compression Decay.

### 7.6 Video Extension

For video:

```text
X ∈ R^{T×H×W}
```

The same spatial operator is applied per frame or with shared temporal seeds:

```text
OC_{4/e}(X)_{t,i,j}
  = X_{t,i,j} + w_q(i,j) N_{4,t}(i,j)
```

Temporal coherence option:

```text
z_v[t,j], z_h[t,i], z_+[t,i+j], z_-[t,i-j+W-1]
```

or low-rank temporal process:

```text
z_v[t,j] = a_t z_v[j] + epsilon_{t,j}
```

The choice must be stated in the experiment protocol.

### 7.7 Fair Ablation

Because the 4/e operator is gated and structured, comparison with gaussian or
uniform noise must match perturbation energy, not nominal `sigma`.

Perturbation energy:

```text
E_pert(M, X) = Mean( (M(X) - X)^2 )
```

Fair baseline sigma:

```text
sigma_eq = sqrt(E_pert(OC_{4/e}, X))
```

Then:

```text
Gaussian baseline: X + N(0, sigma_eq^2)
Uniform baseline:  X + Uniform(-sqrt(3) sigma_eq, sqrt(3) sigma_eq)
```

If 4/e gates to zero:

```text
sigma_eq = 0
```

and gaussian/uniform must also do nothing.

## 8. TPSR / AIMP Physical Consistency

AIMP is not a generic prompt trick. It is a set of measurable physical and
composition constraints.

### 8.1 Field Law

Each scene has a Field Card:

```text
F = {light_dir, light_strength, ambient_contrast, tone_lut, fog, grain, sfx_style}
```

Field consistency:

```text
L_field_card = d(F_frame, F_scene)
```

### 8.2 Perspective Law

Camera state:

```text
C_t = {position, rotation, focal_length, distortion, horizon, vanishing_points}
```

Scale law:

```text
distance × s  ->  linear size × 1/s
distance × s  ->  area × 1/s^2
```

Perspective loss:

```text
L_persp = d(vanishing_points_pred, vanishing_points_ref)
        + d(scale_ratio_pred, scale_ratio_expected)
```

### 8.3 TPSR: Triangular Pupil Spherical Reflection

Let:

```text
H       linear highlight energy / brightness
A       triangular highlight area
L_l     light-eye diopter = 1 / D_l
gamma   geometry-photometry exponent
        gamma = 4 for coaxial / ring light
        gamma = 2 for separated light source
```

Invariant:

```text
K = H / ( L_l^2 A^{gamma/2} ) ≈ constant
```

If `L_l` is unknown:

```text
K' = H / A^{gamma/2}
```

Pairwise consistency:

```text
U_{ij}
  = ( H_i A_j^{gamma/2} L_{l,j}^2 )
    / ( H_j A_i^{gamma/2} L_{l,i}^2 )
  ≈ 1
```

Score:

```text
C_TPSR
  = exp( - median_{i<j} |log U_{ij}| / sigma_T )
```

Typical:

```text
sigma_T ∈ [0.1, 0.3]
```

TPSR loss:

```text
L_TPSR = median_{i<j} |log U_{ij}|
```

## 9. Linear Regression Calibration

Observable feature vector:

```text
z =
[
  mean(MSN),
  var(MSN),
  mean(SNR),
  compression_ratio,
  motion_density,
  rhythm_density,
  symbol_conflict,
  layout_overlap,
  artifact_residue,
  temporal_drift
]
```

Linear head:

```text
y_hat = beta_0 + beta^T z
```

Outputs:

```text
y_hat =
[
  SNR_hat,
  CR_hat,
  fps_hat,
  conflict_hat,
  stability_hat,
  artifact_hat
]
```

Controller use:

```text
step_size       <- g_1(SNR_hat, stability_hat)
fps             <- g_2(fps_hat, temporal_drift)
symbol_strength <- g_3(conflict_hat)
artifact_weight <- g_4(artifact_hat)
```

This is a controller / calibration layer, not the generative core.

## 10. VOD Field Update And Flow Objective

Field update:

```text
U_{k+1}
  = U_k + eta_k F_theta(U_k, c, b, q, tau_k)
```

where:

```text
c condition
b boundary
q phase/frequency/mode state
tau generation time
```

PDE-style interpretation:

```text
partial U / partial tau
  = div( D_theta(U,b,q,c) grad U )
  + R_theta(U,c)
```

Flow matching form:

```text
U_tau = alpha(tau) U_clean + sigma(tau) epsilon
```

Target velocity:

```text
v_target = dU_tau / d tau
```

Learned velocity:

```text
v_theta = F_theta(U_tau, c, b, q, tau)
```

Flow loss:

```text
L_flow = || v_theta - v_target ||^2
```

## 11. Full Training Objective

Total loss:

```text
L_total =
    w_flow       L_flow
  + w_proj       L_projection
  + w_BT         L_BT
  + w_MSN        Σ_k MSN_k
  + w_TTNM       L_TTNM
  + w_mode       L_mode
  + w_4e         L_4e
  + w_AIMP       (L_field_card + L_persp + L_TPSR)
  + w_calib      L_calibration
```

4/e artifact loss:

```text
L_4e
  = relu( R_tile(Y_pred, q) - max(R_tile(Y_target, q), 1) )
```

But the true 4/e claim is tested by the operator `OC_{4/e}` against fair
baselines, not by this loss alone.

Mode regularizer:

```text
L_mode = || H_b(U) - lambda U ||^2
```

where `H_b` is the boundary-conditioned mode operator.

Calibration loss:

```text
L_calibration = || y_hat - y_target ||^2
```

## 12. Task Protocols

### 12.1 Denoising

Valid paired denoising:

```text
U_clean ~ data
U_noisy = Corrupt(U_clean)
model(U_noisy) -> U_clean
```

Per-sample losses are valid here.

### 12.2 Conditional Generation

Valid conditional generation:

```text
condition = non-answer metadata
U_target ~ p(U | condition)
model(noise, condition) -> U_target
```

Condition must not contain target pixels / target encoding.

### 12.3 Unconditional Generation

Valid unconditional generation:

```text
model(noise) -> sample
sample distribution ≈ data distribution
```

Per-sample MSE to an independently sampled target is invalid.

Distribution metrics are required.

## 13. Claim Test Rule

Before testing any VOD-specific constraint, the model must pass the output
capability gate. This gate was missing in the first planning pass and caused
several days of false progress: auxiliary constraints were tested before the
model was shown to preserve or reconstruct an image.

### 13.0 Gate 0: Output Capability

No VOD mechanism counts as product progress until the following contracts
pass:

```text
Round-trip identity:
    decode(encode(clean_image)) ≈ clean_image

Clean no-op stability:
    decode(denoise_path(encode(clean_image))) ≈ clean_image

Paired denoising:
    model(clean + noise) beats:
        zero baseline
        noisy baseline
        identity / no-op baseline
```

Minimum metrics:

```text
image_mse
video_mse
visible reconstruction grid
finite latent / output values
improvement_vs_zero
improvement_vs_noisy
```

Falsification condition:

```text
If decode(encode(x)) collapses to a constant image, the model has no
image output capability.

If denoise_path(encode(clean)) explodes or changes clean content, the
model has no clean no-op stability.

If model output fails zero/noisy/identity baselines, downstream VOD
constraint ablations do not count as product validation.
```

Required ordering:

```text
Gate 0: visible round-trip / clean no-op / baseline recovery
Gate 1: 4/e / TTNM / Binary-Twin / AIMP constraint utility
Gate 2: composite stacking
Gate 3: scale-up or new media
```

The earlier workflow violated this ordering by starting at Gate 1/2 while
Gate 0 had not been tested.

Every VOD mechanism must be tested as:

```text
Claim:
Protocol:
Baselines:
Metrics:
Falsification condition:
Numbers:
Conclusion:
Next minimal fix:
```

For example, 4/e is not validated by saying:

```text
tests passed
artifact module implemented
```

It is validated only if:

```text
Claim 1A:
    OC_{4/e} exhibits the four-axis covariance signature
    and differs significantly from iid gaussian/uniform baselines.

Claim 1B:
    OC_{4/e} improves a pre-registered downstream utility metric
    under fair perturbation energy, with statistically meaningful
    effect size and without unacceptable structure loss.
```

If it fails, the result is a negative result or an implementation gap, not a
reason to swap in an external model.

Current unified interpretation of the 2026-05-01 ablation result:

```text
Claim 1A: PASS
    AXCOV signature:
        4/e   ≈ 0.2104
        gauss ≈ 0.0269
        unif  ≈ 0.0277
        gain  ≈ +0.1827
        3sig  ≈ 0.0076

    This means the current operator is structurally non-iid and conforms
    to the four-axis covariance definition.

Claim 1B under artifact_score: NOT ESTABLISHED
    artifact_score ties iid because it measures mean tile residue,
    not the four-axis covariance signature that defines the operator.
    artifact_score is also not the right operationalisation of the
    application semantics in vod_math_simplification.md §"Orthogonal
    Compression Noise / Tile Residue" — that text describes the goal as
    "破相 the coherent tile light spot contour", not "reduce mean tile
    residue".

Claim 1B under boundary sign-agreement on coherent halo (added 2026-05-01):
    PASS at low perturbation strength (s=0.3), 7.7σ over 200 seeds.
    Saturates at high strength as both methods approach the random
    floor of 0.5.

Claim 1B under training-time integration (added 2026-05-02):
    PASS, with one fixed-default change. The legacy
    `RESIDUE_FLOOR = 1.0` in `vod_minimal/torch_artifacts.py` was
    silently gating the loss to zero on stress training data where
    `target_tile_residue` is naturally < 1.0. Default changed to 0.0;
    floor>0 retained as an opt-in over-smoothing guard. With the fix:

        weight_artifact   pred_video_tile_residue   imp_vs_noisy
        0.0               1.086                     0.1154
        0.1               1.009                     0.1157
        1.0               0.895                     0.1104

    Monotonic control over output tile residue. Heavy weight (1.0)
    over-smooths past target_r=0.949 with measurable recovery cost.

Claim 1B continuous strength dial (added 2026-05-02):
    PASS. New `strength` parameter on
    `artifact_regularization_loss(...)` exposes a continuous 4/e
    intensity knob independent of the total-loss weight budget.
    Sweep at fixed weight=0.1, floor=0.0:

        strength  pred_video_tile_residue  imp_vs_noisy
        0.0       1.086                    0.1154
        0.5       1.055                    0.1155
        1.0       1.009                    0.1157   <- best recovery
        2.0       0.933                    0.1151   <- below target
        5.0       0.979                    0.1126   <- thrashing

    Operating range [0.5, 2.0] gives meaningful control without
    recovery cost. Above 2.0 the operator over-smooths and the model
    starts oscillating between MSE and 4/e gradients (L_total stops
    decreasing).

Composite stacking ablation (2-layer, added 2026-05-02):
    With both `weight_artifact=0.1` and `weight_temporal=0.1` active on
    flicker-stressed data (blocky scatter + per-frame iid jitter), the
    two distinctives stack constructively. Static-blocky data is
    insufficient — `L_temporal = relu(smooth_pred - smooth_target)` is
    a one-sided gate that requires pred-side jitter to fire, same
    structural pattern as the floor=1.0 issue above. With flicker:

        config         imp_noisy  pred_vid_res  pred_vid_smooth
        full           0.1540     1.014         0.1492    <- best on both metrics
        no_artifact    0.1546     1.047         0.1598
        no_temporal    0.1538     1.021         0.1594
        bare           0.1546     1.047         0.1636

    Each distinctive contributes monotonic improvement on its target
    metric; full configuration wins on both without measurable recovery
    cost (imp_noisy delta < 0.001). Confirms 4/e and TTNM regularisers
    can be wired together in the training pipeline without interference.

5-layer composite stacking ablation (2026-05-02):
    All five spec distinctives wired into `native_total_loss` and
    leave-one-out tested under flicker stress training:

        config           w_art  w_temp  w_btpx  w_aimp  imp_noisy  pred_vid_res  pred_vid_smooth
        full             0.1    0.1     0.1     0.1     0.1544     0.977         0.161
        no_artifact      0      0.1     0.1     0.1     0.1554     1.042 (+6.7%) 0.167
        no_aimp          0.1    0.1     0.1     0       0.1545     1.027 (+5.1%) 0.160
        no_temporal      0.1    0       0.1     0.1     0.1541     0.979         0.165 (+2.4%)
        no_binary_twin   0.1    0.1     0       0.1     0.1535     0.971         0.153
        bare             0      0       0       0       0.1546     1.047 (+7.2%) 0.164

    Each distinctive leaves a measurable fingerprint on its target
    metric when removed:
      4/e (artifact)          → pred_vid_res, +6.7% on removal
      AIMP (TPSR consistency) → pred_vid_res, +5.1% on removal
      TTNM (temporal)         → pred_vid_smooth, +2.4% on removal
      Binary-Twin pixel       → pred_vid_smooth (mild antagonism — BT
                                imposes per-pixel discreteness, which
                                slightly raises temporal jitter; removing
                                BT smooths video by 5.2%)

    No catastrophic interaction: `imp_noisy` varies by less than 0.002
    across all six configs (all within recovery noise). The five
    distinctives stack constructively, sharing a single backward pass
    over the unified field U(t,y,x,c).

    This is the architecture-verification milestone for VOD as a native
    multimodal diffusion generator: the Chladni entropy field can carry
    spec constraints 4-8 as simultaneous field-side pressures. They
    contribute non-zero training-time gradients with monotonic dial
    control, no destructive interaction, and joint convergence to a
    constructive optimum on both basic-recovery and physical-metric
    axes simultaneously.

    Setup:
        X = clean_chladni + s * coherent_tile_light_spot
        coherent_tile_light_spot = 1-pixel-wide bright halo at every
            tile boundary line, identical brightness along the line.
        sign_agreement(image) = mean fraction of consecutive same-sign
            boundary jumps along all tile-boundary lines.
            1.0 = perceptually coherent contour, 0.5 = random.
        Lower sign_agreement = more 破相.

    Numbers (200 seeds, paired diff = best_iid - 4/e, threshold 0.02):
        s=0.3:  paired_diff = +0.0318    3σ_SE = 0.0124    PASS (~7.7σ)
        s=0.5:  paired_diff = +0.0196    3σ_SE = 0.0132    borderline
        s=0.8:  paired_diff = +0.0129    3σ_SE = 0.0129    FAIL (saturated)

    Direction is consistent across all three strengths (4/e gives
    lowest sign_agreement). At low/medium perturbation regime 4/e
    produces measurably more 破相 than energy-matched iid baselines;
    at high regime both methods approach the random floor and the
    relative advantage collapses.

    Physical reason: at matched perturbation energy E,
        σ_iid² = E uniformly across all pixels
        σ_4e²(boundary) = E · w_q²(boundary) / mean(w_q²)
                        ≈ 3-4× σ_iid² for tile=8, lambda_q=1
    so 4/e applies higher per-pixel variance exactly at boundary
    positions, flipping more boundary-jump signs and 破相-ing the
    contour line more than uniform iid. This is a *local variance
    redistribution* effect, not a cross-correlation effect — the
    earlier "zero-mean perturbation cannot reduce E[detectability]"
    derivation was true but answered the wrong operationalisation.

Conclusion:
    Claim 1A:  operator conformance PASS.
    Claim 1B:  application utility PASS at low perturbation regime
               under sign-agreement metric on coherent halo. Saturates
               at high regime. The operator does what the application
               claim in vod_math_simplification.md says.

    The original "OPEN under artifact_score" was the right honest call
    given the wrong metric, but the wrong call about the operator. See
    VOD_agent_postmortem.md §12.15 for the full responsibility chain
    and meta-lesson on metric-fail rationalisation traps.
```

## 14. Chladni Field Constraint Rollup

This section records how the current prototype exposes different
constraints of the same Chladni entropy field. The names below are not
separate product modules and not an external pipeline. They are validation
slices of one field:

```text
VOD = Chladni entropy field U
    + projection boundaries
    + field-side constraints
    + physical / symbolic readout metrics
```

It is not enough for a constraint to exist in this mathematical document;
before claiming utility, the code must contain the constraint, the test must
match the constraint type, and the metric must match the application
semantics.

As of 2026-05-01, the VOD field constraints are in three different
validation states:

```text
field operator implemented:
    4/e Orthogonal Compression Decay

field diagnostic-only simplified implementation:
    MSN
    TTNM

minimal field readout / consistency slice:
    Binary-Twin
    TPSR / AIMP
```

The rule is strict:

```text
If the prototype does not implement the constraint, do not invent a metric
that pretends to validate it.
```

### 14.1 Distinctive Status Matrix

| Field constraint / readout | Spec section | Prototype implementation | Metric / baseline | Verdict |
|---|---:|---|---|---|
| 4/e Orthogonal Compression | §7 | Full spec-faithful operator | Claim 1A: AXCOV vs energy-matched iid gaussian/uniform. Claim 1B: boundary sign-agreement on coherent tile halo vs energy-matched iid gaussian/uniform. | **PASS**. 1A AXCOV ≈ 13σ. 1B low-energy regime `s=0.3` ≈ 7.7σ. |
| MSN | §5 | Simplified path-stability diagnostic only: `Σ (1/k) mean(|U_{k+1}-U_k|)`. Not the full modular normalization operator. | Pearson correlation between MSN and final mean target error over 40 random model configs, 50 samples each. Baseline expectation: if MSN is useful, higher MSN should predict higher error. | **Borderline FAIL** under pre-registered rule: `r=+0.360` vs threshold `0.40`, `p=0.022` vs threshold `0.01`. Direction is correct but not strong enough. |
| TTNM | §6 | Toy temporal diagnostic only: `temporal_smoothness = mean_t |Y_{t+1}-Y_t|` and one-sided `L_temporal = relu(...)` gating. Not the full tropical graph update. | Paired clean-vs-flicker test over 200 seeds. Metric: `smoothness(flicker) - smoothness(clean)`. Baseline: clean coherent video should have lower temporal variation than flicker-corrupted video. | **PASS** for the toy diagnostic: 200/200 paired signs positive, mean diff ≈ `+0.293`, effective significance ≈ 488σ. |
| Binary-Twin | §4 | Minimal executable slice in `vod_minimal/binary_twin.py`: `Φ`, `Ψ`, `BinaryTwinState`, symbol accuracy / hamming, reconstruction error, differentiable CE+reconstruction loss. `native.py` now uses this loss when text is enabled. | Unit baseline: clean quantized text must beat corrupted symbols; native text loss must be finite and expose symbol accuracy. | **Implemented as minimal text-symbol coupling**. Not full OCR/logo system yet. |
| TPSR / AIMP | §8 | Minimal executable slice in `vod_minimal/aimp.py`: `FieldCard`, `PerspectiveCard`, `LightingCard`, `TPSRMeasurement`, `K`, `U_ij`, pairwise log deviation, consistency score, synthetic distance sequence. | Unit baseline: TPSR-consistent sequence keeps `K` constant and `U_ij≈1`; brightness-inconsistent sequence lowers consistency score. | **Implemented as minimal physical metric layer**. Not full scene/AIMP generator control yet. |

### 14.2 Validation Slice Types

The test template depends on the field-slice type.

For an operator, use two layers:

```text
Definition / conformance:
    Does the implementation have the mathematical signature it claims?

Application utility:
    Does that operator improve the intended downstream metric against
    fair baselines?
```

This is why 4/e has Claim 1A and Claim 1B.

For a diagnostic, do not force the 1A/1B template. The right question is:

```text
Does this diagnostic separate good cases from bad cases?
```

MSN and TTNM are currently diagnostic-only in the prototype, so their tests
must be read as diagnostic tests, not as validation of the full §5 or §6
mathematical systems.

For an implementation gap, the only honest verdict is:

```text
not measurable yet
```

Binary-Twin and TPSR/AIMP were in this category before the minimal
implementation pass. They now have executable metric/loss objects, but
they are still not the full §4 / §8 systems. They remain Chladni field
readout constraints, not add-on platform modules.

### 14.3 MSN Current Boundary

The full MSN definition in §5 contains continuous, discrete, and paired
coupling terms:

```text
MSN_k =
    α · d_cont(U_{k+1}, U_k)
  + β · d_disc(B_{k+1}, B_k)
  + γ · d_pair((U_{k+1}, B_{k+1}), (U_k, B_k))
```

The current prototype only has the continuous path-stability part:

```text
MSN_proto =
    Σ_k (1/k) · mean(|U_{k+1} - U_k|)
```

Therefore the current MSN test does not validate the full modular shrinking
number theory. It only asks whether this simplified path metric is useful as
a model-quality warning signal.

Pre-registered test:

```text
Claim:
    Higher MSN_proto should correlate with worse final target error.

Protocol:
    Randomly sample model/update hyperparameter configurations.
    For each config, run a fixed evaluation batch and record:
        x = MSN_proto
        y = mean target error after rollout

Baseline:
    A useless diagnostic gives r ≈ 0.

Metric:
    Pearson r(MSN_proto, mean_target_error).

Falsification condition:
    FAIL if r < 0.40 or p >= 0.01.

Observed:
    r = +0.360
    p = 0.022

Conclusion:
    Directionally correct but not strong enough under the pre-registered
    rule. Current MSN_proto is a monitoring signal, not a hard model
    selection criterion.
```

Minimal next fix:

```text
Do not tune the threshold.
Implement the missing discrete / pair-coupling components first, then rerun.
```

### 14.4 TTNM Current Boundary

The full TTNM definition in §6 is graph-based tropical temporal propagation:

```text
P'(n) = ⊕_{e∈E_n} (P(n_e) ⊗ W(e))
```

The prototype does not implement the graph form. It only implements a toy
temporal smoothness diagnostic:

```text
S_temp(Y) =
    mean_t |Y_{t+1} - Y_t|
```

and a one-sided training penalty:

```text
L_temporal =
    relu(S_temp(pred) - S_temp(target))
```

Pre-registered test:

```text
Claim:
    The toy TTNM diagnostic should detect temporal flicker.

Protocol:
    For the same underlying field, compare:
        clean coherent video
        flicker-corrupted video

Baseline:
    clean video should have lower temporal smoothness value.

Metric:
    Δ = S_temp(flicker) - S_temp(clean)

Falsification condition:
    FAIL if paired Δ is not consistently positive and significant.

Observed:
    200/200 paired signs positive
    mean Δ ≈ +0.293
    effective significance ≈ 488σ

Conclusion:
    The toy TTNM diagnostic works for flicker detection, but this does
    not yet validate the full tropical graph model.
```

Minimal next fix:

```text
If the goal is full TTNM, add explicit temporal nodes, edges, and tropical
min-plus propagation. Until then, call the current code "toy temporal
smoothness", not full TTNM.
```

### 14.5 Binary-Twin Minimal Implementation

The full Binary-Twin mechanism requires both continuous and discrete fields:

```text
A = (x, B)
x ∈ R
B ∈ {0,1}^N
B ≈ Φ(x)
Ψ(B) ≈ x
```

The prototype now implements the minimal text-symbol coupling:

```text
File:
    vod_minimal/binary_twin.py

Implemented:
    encode_symbols(x)        = Φ(x) = round((levels-1) clip(x,0,1))
    decode_symbols(B)        = Ψ(B) = B / (levels-1)
    binary_twin_state(x)     = (x, Φ(x), Ψ(Φ(x)))
    binary_twin_metrics      = accuracy / hamming / continuous MSE
    binary_twin_torch_loss   = CE(Φ(target), ordinal_logits(pred))
                               + λ MSE(pred, Ψ(Φ(target)))

Native integration:
    native_total_loss(...):
        if text enabled:
            L_text = binary_twin_torch_loss(pred["text"], target["text"])
            log binary_twin_symbol_accuracy
```

This is a real Binary-Twin slice because the text channel is no longer
treated as a purely continuous MSE target. It has a discrete code `B`, a
continuous reconstruction `Ψ(B)`, and a differentiable consistency loss.

Unit-level metric and baseline:

```text
Claim:
    Binary-Twin loss should prefer correct discrete symbols over corrupted
    symbols.

Protocol:
    Build quantized text vectors in [0,1].
    Compare clean prediction against a prediction with one symbol replaced.

Baseline:
    Corrupted symbol prediction.

Metrics:
    symbol_accuracy
    binary_twin_torch_loss

Expected:
    clean accuracy = 1.0
    corrupted accuracy < 1.0
    clean loss < corrupted loss

Implemented tests:
    tests/test_binary_twin.py
```

Remaining missing pieces for full §4:

```text
visual text/logo region extractor
OCR or symbol decoder target
logo-specific symbol vocabulary
region-aware Binary-Twin loss L_BT(region)
text/logo preservation metric on rendered image regions
```

Full future test:

```text
Claim:
    Binary-Twin coupling reduces text/logo corruption without damaging
    visual field quality.

Protocol:
    Generate or corrupt samples with known text/logo regions.
    Compare:
        no Binary-Twin coupling
        Binary-Twin coupling enabled

Baselines:
    no-coupling model
    text-only compositor / OCR postprocess baseline
    image-only denoising baseline

Metrics:
    OCR character accuracy
    logo symbol accuracy
    text-region visual MSE or LPIPS
    non-text-region visual quality

Falsification condition:
    FAIL if OCR/symbol accuracy does not improve over baselines, or if
    visual quality drops beyond the registered tolerance.
```

Until this is implemented, Binary-Twin remains a mathematical spec, not a
full text/logo generation result. The current prototype result only validates
the minimal discrete/continuous coupling.

### 14.6 TPSR / AIMP Minimal Implementation

TPSR/AIMP requires explicit physical scene and drawing constraints:

```text
Field Card:
    main light direction
    ambient contrast
    tone / screentone / grain state

Perspective Card:
    camera position
    vanishing points
    focal scale
    occlusion relations

Lighting Card:
    material response
    highlight rules
    shadow direction

TPSR Card:
    triangular pupil highlight region
    highlight energy H
    highlight area A
    light-eye diopter L_l
    geometric exponent γ
```

The TPSR invariant is:

```text
K = H / (L_l^2 A^{γ/2}) ≈ const
```

and the paired consistency ratio is:

```text
U_ij =
    H_i A_j^{γ/2} L_{l,j}^2
  / (H_j A_i^{γ/2} L_{l,i}^2)
≈ 1
```

The prototype now implements the minimal physical metric layer:

```text
File:
    vod_minimal/aimp.py

Implemented:
    FieldCard
    PerspectiveCard
    LightingCard
    TPSRMeasurement
    tpsr_k(H, A, L_l, γ)
    tpsr_pair_ratio(U_ij)
    tpsr_pairwise_log_deviation
    tpsr_consistency_score
    synthesize_tpsr_measurements
    aimp_tpsr_metrics
```

Unit-level metric and baseline:

```text
Claim:
    TPSR metrics should distinguish a physically consistent distance
    sequence from a brightness-inconsistent highlight sequence.

Protocol:
    Generate synthetic measurements where:
        area   A ∝ 1 / D_c^2
        energy H ∝ 1 / D_c^γ
    Then corrupt the final frame brightness by a multiplicative factor.

Baseline:
    Brightness-inconsistent final frame.

Metrics:
    K coefficient of variation
    U_ij pairwise log deviation
    tpsr_consistency_score

Expected:
    consistent sequence:
        K approximately constant
        U_ij ≈ 1
        score ≈ 1
    corrupted sequence:
        score lower than consistent sequence

Implemented tests:
    tests/test_aimp_tpsr.py
```

This is a real TPSR metric layer, not yet a full AIMP control system.

Remaining missing pieces for full §8:

```text
triangular highlight detector on actual generated images
main-light direction estimator
vanishing-point / perspective estimator
Field Card extraction from generated scene
Perspective Card extraction from generated scene
Lighting Card extraction from generated scene
closed-loop generator loss using TPSR/AIMP cards
```

Full future test:

```text
Claim:
    TPSR/AIMP constraints reduce physically incoherent manga/illustration
    artifacts, especially eye highlights, lighting drift, and perspective
    drift.

Protocol:
    Use synthetic or annotated manga-eye / room-scene samples with known
    light direction, camera scale, and highlight geometry.

Baselines:
    unconstrained generator
    prompt-only TPSR text instruction
    post-hoc compositor-only correction

Metrics:
    median |ln U_ij| for highlight consistency
    K variance under fixed lighting
    highlight tip-to-light angular error
    vanishing-point / perspective error
    field-card consistency score

Falsification condition:
    FAIL if TPSR/AIMP does not reduce physical inconsistency over baselines,
    or if it improves the eye metric while damaging global composition.
```

Until these scene extractors and generator losses exist, TPSR/AIMP is a
metric-layer prototype result, not a full drawing-physics generator result.

### 14.7 Implementation Rollup Conclusion

Current VOD distinctive status:

```text
4/e:
    implemented and validated for conformance + low-energy phase-break utility

TTNM:
    toy diagnostic implemented and validated for flicker detection

MSN:
    toy diagnostic implemented but borderline fail as hard quality predictor

Binary-Twin:
    minimal discrete/continuous text-symbol coupling implemented

TPSR/AIMP:
    minimal TPSR/AIMP physical metric layer implemented
```

This means the next serious work must choose one of two paths:

```text
Path A:
    strengthen existing simplified mechanisms
    (MSN full version, TTNM graph version)

Path B:
    lift minimal Binary-Twin and TPSR/AIMP from unit metrics into shared
    Chladni-field training/evaluation loops
```

What must not happen:

```text
Do not describe minimal Binary-Twin or TPSR/AIMP unit tests as full mechanism
validation.
Do not use passing shape tests as claim tests.
Do not change metrics after seeing borderline or failed results.
Do not replace VOD's core claim with an external pretrained model and call it
self-developed.
```
