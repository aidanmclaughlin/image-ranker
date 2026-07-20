# Research and system design

## Executive decision

This is a **single-user, inductive preference-learning** problem, not ordinary image classification and not generic aesthetic assessment. The primary observation is now one 1–5 ordinal judgment of one full-screen photograph; legacy pairwise choices remain valuable evidence and keep their historical Elo, but new labeling no longer requires two images at once. Strictly, a single-item score is an ordinal rating or Likert-type item, not a multi-item Likert scale.

The recommended first model is deliberately small:

1. Decode the whole image with a fixed, versioned preprocessing pipeline.
2. Cache a normalized embedding from a frozen vision foundation model.
3. Learn one regularized scalar visual utility with ordered thresholds: cumulative ordinal/CORAL loss consumes 1–5 ratings, while Bradley–Terry loss consumes the immutable legacy comparisons.
4. Use deterministic bootstrap replicas of that same head for uncertainty; this is one model family, not a separate reward model.
5. Use predicted utility only to pre-screen technically valid candidates. Never feed a prediction back as crawler reward.
6. Give the context-free EXP3 source controller only the owner's direct normalized rating, `(rating - 1) / 4`, for the single image attributable to its action.
7. Preserve observed point ratings and historical Elo in the collection rather than fabricating one label type from the other.

The repository currently ships **OpenCLIP ViT-B/32** as the practical baseline because it is open, reproducible, feasible in a bounded CPU Sandbox (and on a local Mac), and already integrated. The research recommendation is not that OpenCLIP is known to be the best aesthetics encoder—there is no paper establishing that for this exact one-person regime—but that it is the correct baseline to beat. DINOv2 and SigLIP/SigLIP 2 should be evaluated on the same image-disjoint rating and comparison folds before any encoder is promoted or combined.

Population aesthetics models, award status, resolution, and source curation are useful **quality priors and acquisition filters**, never substitutes for the user's choices. The personal utility head is trained only from the owner's ratings and legacy comparisons, and the crawler controller is rewarded only by a direct rating. Images, rating and comparison history, embeddings, and model artifacts stay in private deployer-controlled runtime resources and outside Git.

## 1. Problem formulation

For image \(i\), let \(z_i \in \mathbb{R}^d\) be its frozen visual embedding and let \(s_\theta(z_i)\) be the user's predicted scalar utility. A point rating is \(r_i\in\{1,2,3,4,5\}\). Four ordered thresholds \(b_1\leq\cdots\leq b_4\) define cumulative ordinal probabilities

\[
P(r_i>k)=\sigma\left(\frac{s_\theta(z_i)-b_k}{\tau_o}\right),\qquad k=1,\ldots,4.
\]

Training the four cumulative binary targets with shared weights follows the rank-consistent CORAL formulation ([Cao, Mirjalili & Raschka, 2020](https://arxiv.org/abs/1901.07884)). Unlike ordinary five-way classification, it encodes that mistaking a 5 for a 4 is less severe than mistaking it for a 1.

A legacy comparison is \((i,j,y)\), where \(y=1\) means the user chose \(i\) over \(j\). The same scalar utility enters the feature-aware Bradley–Terry model:

\[
P(i \succ j)=\sigma\left(\frac{s_\theta(z_i)-s_\theta(z_j)}{\tau}\right).
\]

The initial utility is linear, \(s_\theta(z)=w^Tz\), with an L2 penalty. Its joint objective is

\[
\mathcal L=\lambda_o\mathcal L_{\mathrm{ordinal}}+
\lambda_p\mathcal L_{\mathrm{BT}}+\lambda_2\lVert w\rVert_2^2,
\]

with each stream normalized before weighting so a large historical comparison table cannot drown out fresh ratings. The ordinal thresholds anchor absolute score levels; the Bradley–Terry term contributes relative directions and is invariant to a shared score offset. This is a single shared utility model trained from two observation types, not two taste models.

This formulation has three useful properties:

- Every point rating locates one image on a stable five-level scale and immediately yields a bounded crawler reward.
- Every retained comparison teaches a direction in feature space; swapping left and right negates the difference and exposes historical position bias.
- Both likelihoods share one utility, so either label type can improve predictions for unseen images without converting one into fake examples of the other.

The target question should stay stable: **“How strongly would I want this photograph in my collection?”** The five anchors need a durable meaning—1 strongly reject, 2 weak, 3 worthwhile, 4 excellent, 5 exceptional—because shifting the interpretation between sessions makes the thresholds ill-defined. Technical validity is a hard pre-model gate, while personal judgment is the learned objective.

One scalar utility assumes preferences are mostly transitive and context independent. Real aesthetic judgments can be cyclic, category dependent, and session dependent; the system should measure those violations rather than immediately fit a more expressive model. Only add context, multiple latent utilities, or time-varying parameters if held-out residuals show a repeatable pattern and there are enough labels to estimate it.

## 2. Ordinal ratings, Bradley–Terry, and Elo

### Why retain the pairwise likelihood

The classical [Bradley–Terry model](https://academic.oup.com/biomet/article-abstract/39/3-4/324/326091) uses logistic random utility; the earlier [Thurstone law of comparative judgment](https://doi.org/10.1037/h0070288) leads, under its common equal-variance case, to a probit link. Both turn latent score differences into pairwise probabilities. Their practical behavior is usually similar at this scale; logistic loss is simpler, stable, and already supported by standard local ML tooling, so Bradley–Terry is the default. A probit challenger is worthwhile only if validation log loss or calibration improves.

If pairwise audits later collect explicit ties, do not convert them randomly into wins. Use a tie-aware likelihood such as [Davidson's extension](https://doi.org/10.1080/01621459.1970.10481082). A skip in the current rating UI means “no usable label,” not a middle score and not a zero reward.

### Two distinct rankings

There are three related but different values:

- **Direct rating:** the owner's observed 1–5 judgment of an image; this is the authoritative new label and crawler reward source.
- **Legacy Elo:** an online summary of immutable historical comparisons, retained for continuity rather than silently translated into point scores.
- **Predictive utility:** the shared visual model's estimate for a new or rated image, used for pre-screening and evaluation but never as policy reward.

They should not be conflated. A feature model can smooth away an idiosyncratic favorite, a five-level score creates real ties, and Elo only compares the historical graph. The collection therefore leads with the observed point rating and preserves Elo for legacy evidence; predicted utility can break analytical ties only when clearly labeled as a prediction.

### Elo's proper role

Elo updated immediately after every old pairwise choice and remains useful historical UI context. It is an online approximation related to Bradley–Terry optimization, but its result depends on presentation order, initial ratings, and the K-factor; recent analysis explicitly studies Elo as online learning under model misspecification ([Tang, Wang & Jin, 2025](https://arxiv.org/abs/2502.10985)). Keep Elo and every comparison immutable, but do not update Elo from a point rating or treat Elo as that rating.

No legacy Elo ranking is trustworthy across disconnected comparison components. The new absolute scale avoids that graph-connectivity requirement for newly rated images, although occasional pairwise audits could still diagnose coarse ties and rating drift later.

## 3. Encoder review

No available benchmark answers “which frozen encoder best predicts one person's taste in high-quality photography from hundreds of personal judgments.” Published transfer results are informative priors, not a substitute for this project's image-disjoint evaluation.

| Encoder | Relevant evidence | Strengths for this project | Risks and decision |
|---|---|---|---|
| **DINOv2** | [Oquab et al.](https://arxiv.org/abs/2304.07193) scale self-supervised learning on a curated, diverse image corpus and report strong frozen transfer at image and pixel levels. | Purely visual representation; strong structural and local features; patch tokens make future multi-scale or composition experiments possible. | It has no text-aligned space, and its published benchmarks are not personal aesthetics. Evaluate a small ViT-B/14 checkpoint as the principal visual-only challenger. |
| **OpenCLIP** | CLIP introduced broad image-language pretraining ([Radford et al.](https://arxiv.org/abs/2103.00020)); OpenCLIP published reproducible training and checkpoints at scale ([Cherti et al.](https://arxiv.org/abs/2212.07143)). | Good semantic/style organization, easy text-guided source exploration, mature open tooling, and a practical ViT-B/32 local baseline. | Web-text supervision can encode popularity, caption, and cultural biases; CLIP similarity is not an aesthetic score. Stock center-crop preprocessing can destroy photographic composition. Keep as v1, not as an assumed winner. |
| **SigLIP / SigLIP 2** | SigLIP replaces global softmax contrastive normalization with a pairwise sigmoid image-text objective ([Zhai et al.](https://arxiv.org/abs/2303.15343)). [SigLIP 2](https://arxiv.org/abs/2502.14786) adds self-supervised and captioning objectives, reports stronger transfer than SigLIP at all released scales, and includes native-aspect-ratio variants. | Strong semantic transfer; newer variants' flexible resolution and aspect-ratio handling are particularly relevant to landscape and panoramic photography. | “Pairwise” here means image-text pretraining, not personal pairwise taste. Local latency, memory, checkpoint license, and integration must be measured. Use as a challenger after the baseline evaluation harness exists. |

### Preprocessing matters as much as the checkpoint

Photographic preference depends on global composition, tonal relationships, and sometimes small detail. A default square center crop can remove the subject or horizon, especially for panoramas and portrait orientation. Each encoder experiment must compare at least:

- a full-frame resize with aspect ratio preserved and padding rather than cropping;
- the checkpoint's canonical preprocessing;
- a higher-resolution or multi-view representation if the first two leave systematic failures.

The cache key must include encoder family, exact weights, weight hash, input resolution, color transform, crop/padding policy, pooling rule, and code version. Changing any of these creates a new representation, not an in-place cache update.

Concatenating DINOv2 and a language-aligned embedding is plausible because the pretraining signals are complementary, but it also doubles dimensionality and overfitting risk. Test each encoder alone first; only promote concatenation if it produces repeatable image-disjoint gains under the same regularization budget.

## 4. What personalized-aesthetics research contributes

Generic image-aesthetics work predicts population opinion, not this user's taste. [NIMA](https://arxiv.org/abs/1709.05424), for example, predicts distributions of human quality/aesthetic ratings and is useful as a weak quality prior. It should not be used to fabricate personal labels or dominate candidate scoring.

The personalized literature supports adaptation from a shared visual representation with a small user-specific component:

- [Ren et al., *Personalized Image Aesthetics*](https://openaccess.thecvf.com/content_iccv_2017/html/Ren_Personalized_Image_Aesthetics_ICCV_2017_paper.html) model a personal residual from generic aesthetics and add active selection, demonstrating that content and aesthetic attributes predict some individual deviation even with limited labels.
- [Lee & Kim, *Image Aesthetic Assessment Based on Pairwise Comparison*](https://openaccess.thecvf.com/content_ICCV_2019/html/Lee_Image_Aesthetic_Assessment_Based_on_Pairwise_Comparison__A_Unified_ICCV_2019_paper.html) show that pairwise comparison can support score regression and personalization through reference images.
- [Yang et al., PARA](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Personalized_Image_Aesthetics_Assessment_With_Rich_Attributes_CVPR_2022_paper.html) collect 31,220 images and 438 subjects and show that both image and subject attributes matter, reinforcing that population taste is not a universal ground truth.
- [Kim et al., 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_Learning_Personalized_Photographic_Style_from_Pairwise_User_Preferences_CVPR_2026_paper.html) study pairwise photographic-style preferences from many users, but their task compares style variants rather than choosing among arbitrary photographs. It is evidence for the efficiency of comparative feedback, not a directly deployable ranking model here.
- [Cao, Mirjalili & Raschka](https://arxiv.org/abs/1901.07884) show how shared-weight cumulative binary tasks and ordered biases produce rank-consistent ordinal predictions; CORAL provides the appropriate small-head loss for the new 1–5 observations.
- A 2026 matched study of expert judgments on Chinese paintings reports that preferences produced more consistent ordinal rankings while ratings anchored an absolute score scale ([Zhao et al., 2026](https://arxiv.org/abs/2605.19776)). Its domain and annotator regime differ from Lumen, but the complementary observation supports retaining both real streams in one utility model rather than manufacturing one label type from the other.

Those systems often exploit many users or a generic population prior. This project has one user and wants the personal signal to remain sovereign. The safest low-data transfer is therefore a frozen general representation plus one small personal utility head. A generic aesthetics score may reject the worst intake tail, but it must be logged separately, ablated during evaluation, and never treated as a personal label or source-policy reward.

## 5. Label efficiency with single-image ratings

The full-screen 1–5 interface reduces the cognitive and display cost of comparing two high-resolution photographs and gives each crawled image an immediately attributable reward. The tradeoff is scale drift, coarser ties, and inconsistent use of the middle categories. Fixed verbal anchors, short sessions, occasional repeated images, and a broad intake distribution matter more than making the model clever early.

Legacy pairwise evidence is still statistically useful. [Maystre & Grossglauser](https://proceedings.mlr.press/v70/maystre17a.html) show that repeated sorting is effective under noisy Bradley–Terry preferences, and [feature-aware BTL theory](https://proceedings.mlr.press/v244/saha24a.html) explains how visual features reduce the comparison burden. The current product nevertheless optimizes new labeling throughput with point ratings; pairwise queries are a possible future audit tool, not a second mandatory workflow.

### Recommended labeling phases

**Cold start: first 50 ratings**

- Seed a deliberately broad pool across subject, color, monochrome, weather, scale, orientation, photographic era, and source—not only known favorites.
- Keep the five category meanings visible in onboarding and stable across sessions; do not reinterpret 3 as a skip.
- Let EXP3 choose sources from the first crawl with an initially uniform distribution and explicit exploration, but do not let an unvalidated visual score narrow the intake.
- Train every five ratings to expose progress and pipeline failures, while treating these early heads as smoke tests rather than reliable taste oracles.

**Early model: roughly 50–500 ratings**

- Preserve randomized source and candidate coverage even as predicted utility begins pre-screening.
- Use bootstrap disagreement and distance from labeled embeddings to keep some unfamiliar candidates, not to invent rewards.
- Consider a small hidden-repeat audit stream to estimate within-person agreement and threshold drift; it must be explicitly modeled because the shipped immutable rating path ordinarily labels each image once.
- Measure discovery yield by eventual direct rating, not by the utility score that selected the image.

These are engineering stages, not sample-complexity guarantees. Effective sample size depends on visual diversity and rating consistency; 100 varied ratings can teach more than 1,000 near-duplicates.

**Maturing model: beyond roughly 500 ratings**

- Adapt pre-screening and exploration using image-disjoint learning curves and online human-rated discovery yield.
- Add source-, photographer-, and embedding-cluster quotas to prevent the model from narrowing the visual world to its early guesses.
- If five-level ties hide meaningful top-order differences, run a small optional pairwise audit among highly rated images and feed those real comparisons to the same utility head.

Skips create no rating, no model label, and no bandit reward. Exact duplicates and obvious near-duplicates should not consume labeling capacity. Missing feedback must remain missing: converting a skip or an unrated imported image to zero would train the crawler to punish latency rather than image quality.

## 6. Uncertainty, diversity, and discovery

For an ordinal prediction spread across adjacent categories, entropy can reflect either **aleatoric uncertainty** (the user genuinely wavers between ratings) or **epistemic uncertainty** (the model lacks evidence). Only the latter is reliably reduced by more labels. Legacy pair probabilities near 0.5 have the same ambiguity.

For the linear head, uncertainty need not require a large neural network. Lumen uses deterministic bootstrap replicas of the same joint ordinal-plus-pairwise head; the replicas are uncertainty samples from one architecture, not separately purposed taste and reward models. Deep ensembles are a well-supported general uncertainty baseline ([Lakshminarayanan, Pritzel & Blundell](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)), but here every member remains small and cheap.

For unseen candidate \(x\), maintain ensemble mean \(\mu(x)\) and standard deviation \(\sigma(x)\). A useful acquisition score is an upper-confidence rule

\[
a(x)=\mu(x)+\beta\sigma(x)+\gamma q(x),
\]

where \(q(x)\) is a separate, bounded generic-quality prior. The queue then applies a diversity penalty such as maximal marginal relevance, subtracting similarity to already-selected candidates. This score can decide which valid image is worth showing; it is never written to the source controller as reward. That separation avoids the reward-model overoptimization failure in which optimization exploits predictor error rather than improving true human value ([Gao et al., 2023](https://proceedings.mlr.press/v202/gao23h.html)).

Uncertainty must be paired with out-of-distribution checks. A candidate far from the labeled embedding support can receive an extreme score for the wrong reason. Record nearest-neighbor distance and source/category novelty, cap model confidence outside supported regions, and route some of those items through the explicit exploration quota.

The discovery metric is not predicted Elo or predicted utility. The model proposes candidates; the user rates them; only that observed rating measures discovery quality and updates the source policy.

## 7. Training schedule and promotion gates

The scheduler retrains after every five ratings through 50 and every 10 ratings thereafter. The five-rating run is a pipeline-validating candidate fit; at 10 ratings, five older images can train the head while the latest five form the first fresh image-disjoint promotion holdout, and after a promotion every subsequent five-rating batch can be evaluated without reusing labels seen by the incumbent. Legacy comparisons retain the earlier cadence: every 20 through 100, then every 50. Either stream becoming due launches one joint run over both streams, pinned to both immutable cutoffs so retries are idempotent. Frequent early runs make progress visible; they do not imply that a five-label model is trustworthy enough to dominate candidate selection.

### Staged capacity

1. **Baseline:** frozen OpenCLIP ViT-B/32, normalized embedding, one L2-regularized scalar utility with CORAL thresholds and a Bradley–Terry legacy term.
2. **Encoder bake-off:** on a fixed split, compare OpenCLIP, DINOv2, and SigLIP/SigLIP 2 with identical head capacity and tuning budget.
3. **Head bake-off:** after about 500–1,000 diverse total labels, compare the linear head with a very small two-layer head and the deterministic bootstrap ensemble under identical joint losses.
4. **Representation adaptation:** only after several thousand diverse ratings and comparisons, test last-block tuning or a parameter-efficient adapter such as [LoRA](https://openreview.net/forum?id=nZeVKeeFYf9). Do not jump directly to full encoder fine-tuning.

Label count alone is not a gate: thousands of repetitive ratings or comparisons from one visual cluster have low effective sample size. Promote a more complex model only when all of the following hold:

- lower image-disjoint ordinal loss/MAE and legacy pairwise log loss with bootstrap confidence intervals that exclude no improvement;
- no material regression in source/photographer holdouts, ordinal calibration, or hidden-repeat behavior;
- gains persist across at least three data splits or temporal folds and random seeds;
- hyperparameters were chosen without touching the final test set;
- hosted CPU embedding/training latency, transfer, and artifact size stay within the product budget.

Every artifact should store both rating and comparison cutoffs and counts, split IDs, encoder and weight hash, preprocessing manifest, random seed, hyperparameters, source commit, and per-stream metrics. Retraining should create immutable versioned artifacts and switch the active model only after gates pass, so rollback is exact.

## 8. Acquisition and crawler design

The crawler is a staged acquisition system, not a general-purpose web scraper:

```text
provider API -> rights/provenance gate -> download/decode -> technical-quality gate
             -> exact/perceptual dedupe -> embedding + weak quality prior
             -> optional taste pre-screen + diversity quotas -> one-photo queue
             -> direct 1–5 rating -> source reward + joint retraining
```

### Provider contract

Each source adapter must emit, before or alongside bytes:

- provider name and stable provider item ID;
- canonical work URL and exact asset URL;
- creator, title, date, and source collection when available;
- license identifier, license URL, public-domain flag, attribution text, and retrieval time;
- original dimensions, MIME type, file checksum, and provider metadata snapshot.

Unknown or incompatible rights block automated download by default. An LLM must never infer a license from page prose or search snippets.

### Cheap filters before model scoring

- require a successful full decode and allowlisted JPEG, PNG, or WebP MIME/type agreement;
- reject truncated, zero-area, animated, or implausibly shaped assets;
- enforce the current floor of 1,200 px on the shorter edge and 2.5 megapixels, while keeping both thresholds configurable;
- use exact SHA-256 plus perceptual/embedding deduplication across resized, recompressed, and cropped copies;
- retain the highest-resolution rights-compatible copy and merge provenance records instead of creating duplicate items.

Resolution is necessary but not sufficient. Blur, low contrast, darkness, grain, monochrome, and unusual aspect ratio can all be intentional artistic choices, so aggressive “quality” thresholds risk deleting exactly the work the system is meant to discover. Use conservative corruption checks and provider editorial signals as hard gates; treat learned generic aesthetic or no-reference-quality scores only as soft priors.

### Intelligent exploration

Source metadata and text-aligned embeddings can expand queries from the user's seeds: National Geographic-style documentary landscape, classic large-format monochrome, award-winning landscape, wildlife, night sky, aerial, desert, mountain, forest, ocean, and so on. LLM agents may propose source queries, categories, and photographer names, then audit coverage. They should not fetch arbitrary image-search results or write unverified files directly into the collection.

The scheduler should maintain explicit budgets for exploitation, uncertainty, random exploration, new sources, underrepresented embedding clusters, and underrepresented creators. Without those budgets, a taste model used for pre-screening creates a feedback loop: it sees more of what it already understands, becomes more confident there, and mistakes narrowness for taste. Direct human reward prevents prediction from certifying its own choices, but it does not by itself fix selective exposure.

## 9. Direct-reward source controller

The crawler's controller is intentionally a **context-free multi-armed bandit**, not a second vision model and not deep reinforcement learning. Each arm is a rights-clean Wikimedia source category; an action chooses which category's next API page to inspect; and only the eventual outcome of that action is observed. Advancing an opaque continuation token is persistent operational state, but the policy does not consume state features or optimize a validated long-horizon return. Calling the shipped controller contextual RL would overstate what it learns.

The controller learns only **which source categories yield highly rated imported images**. The shared ordinal-plus-pairwise vision head can rank technically valid candidates within a fetched pool to conserve labeling attention, but its score never becomes action reward. Rights, decode, resolution, byte, corruption, and deduplication checks remain hard gates outside the policy, so reward optimization cannot trade them away.

### Why discounted EXP3-IX is the current choice

The source problem has a small discrete action set, bandit-only feedback, very few daily actions, changing provider contents, and delayed human ratings. These properties favor a small adversarial bandit over a high-capacity stochastic controller. EXP3 supplies the exponential-weights foundation for non-stochastic rewards ([Auer et al., 2002](https://www.schapire.net/papers/AuerCeFrSc01.pdf)); EXP3-IX stabilizes importance-weighted feedback through implicit exploration ([Neu, 2015](https://proceedings.neurips.cc/paper/2015/hash/e5a4d6bf330f23a8707bb0d6001dfbe8-Abstract.html)); and discounting emphasizes recent evidence as taste, category contents, and intake filters drift ([Garivier & Moulines, 2011](https://arxiv.org/abs/0805.3415)). The shipped discounted EXP3-IX policy is an engineering combination of those ideas, not a claim that their regret theorems transfer unchanged to this delayed, non-stationary product loop.

For available categories \(A_t\), the worker replays at most 4,096 completed observations. Before each replayed observation it discounts every log weight by \(0.995\). With

\[
\eta_t=\min\left(0.25,\sqrt{\frac{\log K}{K t}}\right),\qquad
\gamma_t=\eta_t/2,
\]

the chosen arm receives the bounded reward estimate \(\eta_t r_t/(p_t+\gamma_t)\). The next source distribution is a softmax over those log weights mixed with 20% uniform exploration:

\[
p_t(a)=0.8\,\frac{\exp(w_a)}{\sum_{j\in A_t}\exp(w_j)}
       +\frac{0.2}{|A_t|}.
\]

The explicit 20% mixture is deliberately retained even though IX already regularizes the estimator: it guarantees continuing source coverage, bounds every available arm away from zero, and creates overlap for later counterfactual evaluation. The first crawl therefore begins uniformly without waiting for any vision model. Exhausted categories are removed from that round's available set and probabilities are renormalized over the rest.

### Direct, attributable reward

Each source action may import at most one photograph, giving exact credit assignment between an arm choice and a later human judgment. If that photograph receives rating \(r\in\{1,2,3,4,5\}\), the controller observes

\[
r_{\mathrm{human}}=\frac{r-1}{4}\in\{0,0.25,0.5,0.75,1\}.
\]

The mapping preserves the ordering and endpoints of the owner's scale without claiming equal psychological distance between categories; EXP3 only needs a bounded numerical reward. A fully evaluated source action that finds no importable image receives zero because it consumed the opportunity and produced no item. An action with an imported but unrated image remains pending, not zero. A skip likewise remains missing. A genuinely truncated/censored fetch and an operational failure are logged but excluded from reward replay because their outcomes were not fully observed.

Attribution is deliberately narrower than maximizing a predicted score over a whole page. One action, one imported image, and one immutable rating make the reward understandable and auditable. The cost is slower policy feedback and lower crawl throughput, an appropriate trade for a single user whose attention—not API volume—is the scarce resource.

### Delayed human feedback

The selected source action is recorded immediately, but its imported image may be rated hours or days later. The rating transaction atomically links the normalized reward back to that action; the next policy replay incorporates it. No provisional model score is filled in while the action waits, and no later Elo, anchor comparison, or generic aesthetic score replaces the direct observation. Delayed bandit feedback is a distinct statistical problem ([Joulani, György & Szepesvári, 2013](https://proceedings.mlr.press/v28/joulani13.html)), so policy learning curves must be indexed by observed rewards rather than merely actions launched.

Policy versions partition incompatible reward definitions. Historical rows created by the retired model-scored scheme remain audit data but do not enter direct-rating policy replay; otherwise the controller would combine quantities with different meanings and manufacture apparent sample size.

### Exact logging, exploration, and off-policy evaluation

Every source action durably records the selected arm, its exact behavior propensity, the complete probability map over the then-available arms, policy version, optional model-run ID used for pre-screening, action index, outcome status, candidate counts, the one imported image ID, and its eventual direct human/effective reward. Failed, censored, and still-pending actions are retained for audit but excluded from reward replay. This makes the behavior policy reconstructible and, because every available category has positive probability, provides the support needed for inverse-propensity, self-normalized, or doubly robust evaluation. Propensity logging is necessary but not sufficient: offline estimates still need adequate overlap, controlled variance, and one fixed reward definition ([Li et al., 2011](https://arxiv.org/abs/1003.5956); [Dudík, Langford & Li, 2011](https://arxiv.org/abs/1103.4601); [Wang, Agarwal & Dudík, 2017](https://proceedings.mlr.press/v70/wang17a.html)).

No off-policy estimator is promoted automatically yet. With this single user's small stream, policy comparisons must wait for useful effective sample size, remain within the direct-reward policy version, report clipped-weight sensitivity and confidence intervals, and use observed human ratings. The source logs support that future evaluation; they do not make a handful of actions conclusive.

Source exploration and candidate exploration are separate controls:

- **20% source exploration** is truly randomized and propensity-logged in the behavior distribution above.
- **Candidate exploration** preserves some candidates outside the visual model's top pre-screened prefix. Unless its exact probability is logged, it protects diversity but does not support candidate-level off-policy claims.

### Why not PPO, DQN, or NeuralUCB yet

| System | What the primary work assumes or provides | Why it is not the current controller |
|---|---|---|
| **PPO** | An on-policy policy-gradient method that alternates environment sampling with multiple epochs on a clipped surrogate objective ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)). | Five imports per day cannot supply the volume of on-policy trajectories a neural policy needs, and the present crawl has no validated long-horizon return for PPO to optimize. |
| **DQN** | Value learning from replayed transitions in an MDP, demonstrated at scale on Atari frames and game rewards ([Mnih et al., 2015](https://www.nature.com/articles/nature14236)). | Lumen currently observes one source page's bandit reward, not dense reusable state-transition experience; bootstrapped Q-values would add severe sample and representation error. |
| **NeuralUCB** | A neural contextual bandit with UCB exploration under a bounded stochastic reward-function setup ([Zhou, Li & Gu, 2020](https://proceedings.mlr.press/v119/zhou20a.html)). | It becomes attractive when thousands of changing links can share stable context features, but today there are few named category arms, little feedback, and non-stationarity from both taste-model updates and moving provider contents. |
| **Discounted EXP3-IX** | Context-free adversarial source bandit with implicit importance regularization, explicit coverage, and recency weighting. | It matches the sparse direct ratings actually available, is cheap enough to replay exactly, exposes every action probability, and can be replaced only after logged evidence supports a richer policy. |

### Future: sleeping, contextual link-frontier learning

The next justified expansion is not immediately deep RL; it is a **sleeping contextual bandit** over frontier links. Individual categories, providers, searches, and links appear, expire, exhaust, or become temporarily unavailable—precisely the changing-action-set setting studied by sleeping bandits ([Kanade, McMahan & Bryan, 2009](https://proceedings.mlr.press/v5/kanade09a.html); [Saha, Gaillard & Valko, 2020](https://proceedings.mlr.press/v119/saha20a.html)). Each available link should carry pre-fetch context such as provider and category, frontier depth, rights confidence, text/CLIP query embedding, creator novelty, recent domain yield, estimated bytes, and request cost. A linear contextual policy should be the first challenger; NeuralUCB is warranted only if nonlinear held-out gains justify its capacity.

That frontier must log the full available action set, feature version, selected-link propensity, fetch cost, continuation mutation, censoring, and delayed direct ratings. Offline IPS/SNIPS/DR estimates and a shadow replay should beat discounted EXP3-IX on human-rated discovery yield before promotion. Only if link choices demonstrably change valuable future reachable states—and enough complete trajectories exist to learn that effect—should the crawler be upgraded from a contextual bandit to an MDP and benchmark PPO, DQN, or a graph-specific RL method.

## 10. Evaluation and leakage control

### Three questions require three splits

1. **Future judgments on known images:** a rolling chronological or deliberately repeated-rating holdout may contain images seen earlier. This measures within-person stability and threshold drift, but not generalization to new photographs.
2. **Taste prediction for new images:** an image-disjoint split must keep every image, its rating, and every comparison incident to it out of training. This requires planning evaluation pools prospectively; randomly splitting rows or comparison edges is not image-disjoint.
3. **Discovery beyond familiar sources:** source-, photographer-, and near-duplicate-cluster holdouts test whether the model learned visual taste rather than source identity or a photographer signature.

Use validation folds for encoder, regularization, acquisition, and calibration choices, then evaluate once on the locked test set. Group bootstrap resampling by image cluster or labeling session rather than pretending comparison edges are independent.

### Metrics

- **Primary ordinal metrics:** cumulative log loss, mean absolute error, quadratic-weighted kappa, and per-threshold calibration on held-out ratings.
- **Legacy pairwise metrics:** log loss, accuracy, Brier score, and reliability on retained comparisons.
- **Ranking quality:** Spearman correlation and NDCG over sufficiently broad held-out ratings, with Kendall correlation against pairwise audit rankings when available.
- **Label efficiency:** joint learning curves versus number and mixture of ratings/comparisons, including ratings-only and legacy-only ablations.
- **Discovery yield:** mean direct normalized reward, fraction of proposed candidates rated 4 or 5, creator/source novelty, embedding diversity, delay-to-rating, and reward per crawl action.
- **Human consistency:** repeated-rating agreement, threshold usage, skip rate, session-duration effects, and historical left/right bias. This is a noise ceiling, not a model score.

### Common leakage paths

- the same photograph as a thumbnail, crop, recompression, monochrome conversion, or mirror across train and test;
- a burst or near-identical series from one photographer across splits;
- one image's rating in training while a comparison containing that image is in validation, or the reverse;
- repeated comparisons of the same pair with only side order changed;
- provider awards, popularity, favorites, captions, or photographer names passed into the taste head;
- tuning encoder, preprocessing, or thresholds after inspecting the locked test set;
- evaluating only candidates selected by the current model, which hides misses outside its feedback loop.

Deduplicate before splitting, keep metadata out of the personal visual head, maintain a random audit stream, and record all split logic in the private model manifest.

## 11. Chosen v1 design and tradeoffs

| Decision | v1 choice | Reason | Revisit when |
|---|---|---|---|
| Personal objective | One scalar utility with CORAL ordinal loss plus legacy Bradley–Terry loss | Learns from every real label without maintaining separate taste/reward models or synthesizing labels. | Per-stream ablations show negative transfer or stable category-conditioned failures. |
| Encoder | Frozen OpenCLIP ViT-B/32 | Already integrated, CPU-feasible, reproducible, and a credible semantic baseline. | The controlled DINOv2/SigLIP bake-off has enough disjoint labels. |
| Collection rank | Direct 1–5 rating first, historical Elo retained | Keeps observed evidence interpretable and preserves the existing comparison history. | Top-rating ties justify optional pairwise audits or a clearly labeled predictive tie-break. |
| Cold start | Diverse editorial/open-access seeds plus uniform EXP3 source probabilities | Coverage matters more than premature model confidence; the source controller needs no model gate. | Direct reward history supports non-uniform source weights. |
| Candidate selection | Utility/uncertainty pre-screen plus explicit exploration | Conserves attention while keeping predictive output outside the reward loop. | Logged random-audit ablations show a better allocation. |
| Generic aesthetics | Soft intake prior only | Removes obvious low-value tail without overwriting personal taste. | Its incremental discovery yield is proven on random audits. |
| Fine-tuning | Frozen encoder | Hundreds of labels cannot safely support millions of adapted weights. | Several thousand diverse labels and all promotion gates pass. |
| Crawling | Rights-explicit adapters plus context-free discounted EXP3-IX | Learns source yield directly from `(rating - 1) / 4` with exact action propensities and no predicted reward. | Logged direct-reward OPE supports a sleeping contextual link policy. |

## 12. Rights-aware sourcing

Rights metadata is part of model data integrity, not an afterthought. A URL, public accessibility, award, social-media post, or `robots.txt` allowance does not grant copyright permission. The [Robots Exclusion Protocol](https://www.rfc-editor.org/rfc/rfc9309.html) controls crawler access; it is not a content license.

Recommended sources, in priority order:

- **Wikimedia Commons Featured, Quality, and Picture of the Year images.** Use the official [Imageinfo API](https://www.mediawiki.org/wiki/API:Imageinfo), store `extmetadata`, and evaluate every file's license individually using the [Commons reuse guide](https://commons.wikimedia.org/wiki/Commons:Simple_media_reuse_guide). Category membership is a quality prior, not a license field.
- **The Metropolitan Museum of Art.** The official [Collection API](https://metmuseum.github.io/) exposes `isPublicDomain` and corresponding high-resolution images; ingest only records whose asset is actually public domain.
- **Smithsonian Open Access.** Use only assets explicitly designated CC0 through its [developer tools and API](https://www.si.edu/openaccess/devtools); the [FAQ](https://www.si.edu/openaccess/faq) notes that third-party rights may still exist.
- **Cleveland Museum of Art.** Its official [Open Access API](https://www.clevelandart.org/open-access-api) provides CC0 data and rights-compatible image assets.
- **Art Institute of Chicago.** Its [Open Access program](https://www.artic.edu/open-access) and [API](https://api.artic.edu/docs/) expose public-domain status and IIIF assets. Require `is_public_domain` rather than assuming the whole collection is unrestricted.

Unsplash is not a drop-in source for the private archive. Its official [API documentation](https://unsplash.com/documentation) and [guidelines](https://help.unsplash.com/en/articles/2511245-unsplash-api-guidelines) require hotlinking, attribution in API displays, and download-event tracking. Add it only through a purpose-built adapter that follows the current terms; do not silently cache its API images as ordinary stored files.

National Geographic, Instagram, photography competitions, and photographers' own sites are excellent inspiration and discovery metadata but should not be automatically downloaded without an official compatible API, an explicit per-work license, or direct permission. For such sources, support a metadata-only link-out or a clearly marked manual user import; never redistribute those files with the OSS repository or model artifacts.

For every displayed image, preserve creator, title, source link, license name/link, and any required attribution. CC0 still benefits from provenance. CC BY and CC BY-SA require compliance with their exact terms; do not normalize distinct licenses into a generic “free” flag. This is engineering guidance, not legal advice.

## 13. Primary-source reading list

### Preference and ranking

- Thurstone, [*A Law of Comparative Judgment*](https://doi.org/10.1037/h0070288) (1927).
- Bradley & Terry, [*Rank Analysis of Incomplete Block Designs: The Method of Paired Comparisons*](https://academic.oup.com/biomet/article-abstract/39/3-4/324/326091) (1952).
- Davidson, [*On Extending the Bradley–Terry Model to Accommodate Ties*](https://doi.org/10.1080/01621459.1970.10481082) (1970).
- Cao, Mirjalili & Raschka, [*Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation*](https://arxiv.org/abs/1901.07884) (Pattern Recognition Letters, 2020).
- Zhao et al., [*Preferences Order, Ratings Anchor: From Fused Expert Aesthetic Ground Truth to Self-Distillation*](https://arxiv.org/abs/2605.19776) (2026).
- Maystre & Grossglauser, [*Just Sort It! A Simple and Effective Approach to Active Preference Learning*](https://proceedings.mlr.press/v70/maystre17a.html) (ICML 2017).
- Saha & Rajkumar, [*A Graph Theoretic Approach for Preference Learning with Feature Information*](https://proceedings.mlr.press/v244/saha24a.html) (UAI 2024).
- Tang, Wang & Jin, [*Is Elo Rating Reliable? A Study Under Model Misspecification*](https://arxiv.org/abs/2502.10985) (2025).

### Visual representation and aesthetics

- Radford et al., [*Learning Transferable Visual Models From Natural Language Supervision*](https://arxiv.org/abs/2103.00020) (CLIP, 2021).
- Cherti et al., [*Reproducible Scaling Laws for Contrastive Language–Image Learning*](https://arxiv.org/abs/2212.07143) (OpenCLIP, 2023).
- Oquab et al., [*DINOv2: Learning Robust Visual Features without Supervision*](https://arxiv.org/abs/2304.07193) (2023).
- Zhai et al., [*Sigmoid Loss for Language Image Pre-Training*](https://arxiv.org/abs/2303.15343) (SigLIP, ICCV 2023).
- Tschannen et al., [*SigLIP 2*](https://arxiv.org/abs/2502.14786) (2025).
- Talebi & Milanfar, [*NIMA: Neural Image Assessment*](https://arxiv.org/abs/1709.05424) (2018).

### Personalization and label efficiency

- Ren et al., [*Personalized Image Aesthetics*](https://openaccess.thecvf.com/content_iccv_2017/html/Ren_Personalized_Image_Aesthetics_ICCV_2017_paper.html) (ICCV 2017).
- Lee & Kim, [*Image Aesthetic Assessment Based on Pairwise Comparison*](https://openaccess.thecvf.com/content_ICCV_2019/html/Lee_Image_Aesthetic_Assessment_Based_on_Pairwise_Comparison__A_Unified_ICCV_2019_paper.html) (ICCV 2019).
- Yang et al., [*Personalized Image Aesthetics Assessment With Rich Attributes*](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Personalized_Image_Aesthetics_Assessment_With_Rich_Attributes_CVPR_2022_paper.html) (CVPR 2022).
- Kim, Yoo & Kim, [*Learning Personalized Photographic Style from Pairwise User Preferences*](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_Learning_Personalized_Photographic_Style_from_Pairwise_User_Preferences_CVPR_2026_paper.html) (CVPR 2026).
- Houlsby et al., [*Bayesian Active Learning for Classification and Preference Learning*](https://arxiv.org/abs/1112.5745) (2011).
- Biyik & Sadigh, [*Batch Active Preference-Based Learning of Reward Functions*](https://proceedings.mlr.press/v87/biyik18a.html) (CoRL 2018).
- Lakshminarayanan, Pritzel & Blundell, [*Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles*](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) (NeurIPS 2017).
- Hu et al., [*LoRA: Low-Rank Adaptation of Large Language Models*](https://openreview.net/forum?id=nZeVKeeFYf9) (ICLR 2022); relevant as an adaptation mechanism, not evidence that vision-encoder tuning will help this task.

### Bandits, reinforcement learning, and counterfactual evaluation

- Auer et al., [*The Nonstochastic Multiarmed Bandit Problem*](https://www.schapire.net/papers/AuerCeFrSc01.pdf) (SIAM Journal on Computing, 2002).
- Neu, [*Explore No More: Improved High-Probability Regret Bounds for Non-Stochastic Bandits*](https://proceedings.neurips.cc/paper/2015/hash/e5a4d6bf330f23a8707bb0d6001dfbe8-Abstract.html) (NeurIPS 2015).
- Garivier & Moulines, [*On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems*](https://arxiv.org/abs/0805.3415) (ALT 2011).
- Joulani, György & Szepesvári, [*Online Learning under Delayed Feedback*](https://proceedings.mlr.press/v28/joulani13.html) (ICML 2013).
- Gao et al., [*Scaling Laws for Reward Model Overoptimization*](https://proceedings.mlr.press/v202/gao23h.html) (ICML 2023).
- Li et al., [*A Contextual-Bandit Approach to Personalized News Article Recommendation*](https://arxiv.org/abs/1003.0146) (WWW 2010).
- Kanade, McMahan & Bryan, [*Sleeping Experts and Bandits with Stochastic Action Availability and Adversarial Rewards*](https://proceedings.mlr.press/v5/kanade09a.html) (AISTATS 2009).
- Saha, Gaillard & Valko, [*Improved Sleeping Bandits with Stochastic Action Sets and Adversarial Rewards*](https://proceedings.mlr.press/v119/saha20a.html) (ICML 2020).
- Zhou, Li & Gu, [*Neural Contextual Bandits with UCB-Based Exploration*](https://proceedings.mlr.press/v119/zhou20a.html) (ICML 2020).
- Li et al., [*Unbiased Offline Evaluation of Contextual-Bandit-Based News Article Recommendation Algorithms*](https://arxiv.org/abs/1003.5956) (WSDM 2011).
- Dudík, Langford & Li, [*Doubly Robust Policy Evaluation and Learning*](https://arxiv.org/abs/1103.4601) (ICML 2011).
- Wang, Agarwal & Dudík, [*Optimal and Adaptive Off-Policy Evaluation in Contextual Bandits*](https://proceedings.mlr.press/v70/wang17a.html) (ICML 2017).
- Mnih et al., [*Human-Level Control through Deep Reinforcement Learning*](https://www.nature.com/articles/nature14236) (Nature, 2015).
- Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347) (2017).
