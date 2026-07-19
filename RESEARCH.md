# Research and system design

## Executive decision

This is a **single-user, inductive preference-learning** problem, not ordinary image classification and not generic aesthetic assessment. The system must learn a function that assigns a latent utility to a previously unseen photograph from a small number of noisy pairwise choices, while separately maintaining an interpretable ranking of photographs the user has actually judged.

The recommended first model is deliberately small:

1. Decode the whole image with a fixed, versioned preprocessing pipeline.
2. Cache a normalized embedding from a frozen vision foundation model.
3. Learn one regularized scalar utility function from embedding differences with a Bradley–Terry logistic loss.
4. Use that function only to predict personal taste and to choose informative candidates.
5. Keep live Elo for instant UI feedback, but periodically refit an item-only Bradley–Terry model over the complete comparison graph for the canonical ranking.

The repository currently ships **OpenCLIP ViT-B/32** as the practical baseline because it is open, reproducible, feasible in a bounded CPU Sandbox (and on a local Mac), and already integrated. The research recommendation is not that OpenCLIP is known to be the best aesthetics encoder—there is no paper establishing that for this exact one-person regime—but that it is the correct baseline to beat. DINOv2 and SigLIP/SigLIP 2 should be evaluated on the same image-disjoint comparisons before any encoder is promoted or combined.

Population aesthetics models, award status, resolution, and source curation are useful **quality priors and acquisition filters**, never substitutes for the user's choices. The personal preference model must be trained only from the user's labels. Images, comparison history, embeddings, and model artifacts stay in private deployer-controlled runtime resources and outside Git.

## 1. Problem formulation

For image \(i\), let \(z_i \in \mathbb{R}^d\) be its frozen visual embedding and let \(s_\theta(z_i)\) be the user's predicted utility. A comparison is \((i,j,y)\), where \(y=1\) means the user chose \(i\) over \(j\). The feature-aware Bradley–Terry model is

\[
P(i \succ j)=\sigma\left(\frac{s_\theta(z_i)-s_\theta(z_j)}{\tau}\right).
\]

The initial utility is linear, \(s_\theta(z)=w^Tz\), trained with binary cross-entropy and L2 regularization. The intercept is omitted because it cancels in every score difference. The temperature \(\tau\), or an equivalent scale constraint on \(w\), is needed because pairwise data identify relative location and scale only up to the model's convention.

This formulation has three useful properties:

- Every label teaches a direction in feature space, so a choice between two known images can improve predictions for unseen images.
- Swapping left and right negates the input difference, making symmetry explicit and exposing position bias when it occurs.
- The model yields a probability, so log loss, calibration, uncertainty, and active-query policies are all well defined.

The target question should stay stable: **“Which photograph would I rather keep in my collection?”** Mixing “technically best,” “most important,” “most original,” and “my favorite” within the same label makes the latent utility ill-defined. Technical validity is therefore a hard pre-model gate, while personal choice is the learned objective.

One scalar utility assumes preferences are mostly transitive and context independent. Real aesthetic judgments can be cyclic, category dependent, and session dependent; the system should measure those violations rather than immediately fit a more expressive model. Only add context, multiple latent utilities, or time-varying parameters if held-out residuals show a repeatable pattern and there are enough labels to estimate it.

## 2. Bradley–Terry, Thurstone, and Elo

### Bradley–Terry and Thurstone–Mosteller

The classical [Bradley–Terry model](https://academic.oup.com/biomet/article-abstract/39/3-4/324/326091) uses logistic random utility; the earlier [Thurstone law of comparative judgment](https://doi.org/10.1037/h0070288) leads, under its common equal-variance case, to a probit link. Both turn latent score differences into pairwise probabilities. Their practical behavior is usually similar at this scale; logistic loss is simpler, stable, and already supported by standard local ML tooling, so Bradley–Terry is the default. A probit challenger is worthwhile only if validation log loss or calibration improves.

If the UI later collects explicit ties, do not convert them randomly into wins. Use a tie-aware likelihood such as [Davidson's extension](https://doi.org/10.1080/01621459.1970.10481082). A skip means “no usable label,” not a tie.

### Two distinct rankings

There are two related but different estimands:

- **Collection rank:** how the already-labeled images compare, estimated from item identities and their comparison graph.
- **Predictive taste:** how likely the user is to prefer a new image, estimated from visual features.

They should not be conflated. A feature model can smooth away an idiosyncratic favorite, while an item-only model cannot score unseen candidates. The long-run canonical collection rank should therefore be a regularized batch Bradley–Terry fit over item parameters; the crawler and pair selector should use the feature model. The shipped v1 deliberately displays live Elo, as requested, and retains every comparison so that batch refitting can be introduced once the graph has enough coverage to make it meaningful.

### Elo's proper role

Elo updates a displayed rating immediately after each choice, making it excellent UI feedback. It is also an online approximation related to Bradley–Terry optimization, but its result depends on presentation order, initial ratings, and the K-factor; recent analysis explicitly studies Elo as online learning under model misspecification ([Tang, Wang & Jin, 2025](https://arxiv.org/abs/2502.10985)). Keep Elo as a live estimate, store every comparison immutably, and rebuild canonical scores from the entire graph.

No item ranking is trustworthy across disconnected comparison components. Pair selection must schedule bridge comparisons, under-compared images, and occasional anchor comparisons so the graph stays connected. Regularization stabilizes sparse items but does not manufacture missing evidence.

## 3. Encoder review

No available benchmark answers “which frozen encoder best predicts one person's taste in high-quality photography from hundreds of comparisons.” Published transfer results are informative priors, not a substitute for this project's image-disjoint evaluation.

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

Those systems often exploit many users, absolute ratings, or a generic population prior. This project has one user and wants the personal signal to remain sovereign. The safest low-data transfer is therefore a frozen general representation plus a small personal head. A generic aesthetics score may rank the unlabeled intake queue or reject the worst tail, but it should be logged as a separate feature and ablated during evaluation.

## 5. Label efficiency and active preference learning

Random pairs waste labels when images are obviously different; pure uncertainty sampling also wastes labels by repeatedly asking genuine near-ties or out-of-distribution pairs. The acquisition policy needs information, coverage, and diversity.

[Maystre & Grossglauser](https://proceedings.mlr.press/v70/maystre17a.html) show that repeated sorting is a simple, effective active strategy under noisy Bradley–Terry preferences. [Feature-aware BTL theory](https://proceedings.mlr.press/v244/saha24a.html) shows why item features can reduce the comparison burden relative to learning every item independently. [BALD](https://arxiv.org/abs/1112.5745) provides an information-theoretic distinction between predictive uncertainty and information about model parameters, while [batch active preference learning](https://proceedings.mlr.press/v87/biyik18a.html) motivates removing redundant queries from a batch.

### Recommended labeling phases

**Cold start: first 100–200 choices**

- Seed a deliberately broad pool across subject, color, monochrome, weather, scale, orientation, photographic era, and source—not only known favorites.
- Use repeated noisy sorting, under-compared images, and graph-bridging pairs; do not let an unvalidated model control the session.
- Randomize left/right placement. Make roughly 5–10% of trials hidden repeats, usually with sides reversed, to estimate consistency and position bias.
- Include a few easy anchor comparisons among difficult pairs. They help detect fatigue and keep the graph calibrated.

**Early model: roughly 200–1,000 choices**

- 50% near-tie predictive entropy, subject to diversity constraints.
- 25% epistemic disagreement among bootstrapped heads.
- 15% graph repair: under-compared images, disconnected regions, and stable anchors.
- 10% random exploration from the rights- and quality-clean pool.

These percentages are engineering starting points, not claims from a paper. Log the acquisition reason for every pair and ablate the mixture against random and repeated-sort baselines.

**Maturing model: beyond roughly 1,000 choices**

- Adapt the mixture using held-out learning curves and online discovery yield.
- Add source-, photographer-, and embedding-cluster quotas to prevent the model from narrowing the visual world to its early guesses.
- Revisit old top images against new high-potential candidates, but cap repeats so already popular items do not consume the labeling budget.

Skips create no preference label. Exact duplicates and obvious near-duplicates should never be compared. If the user repeatedly cannot choose between genuinely distinct images, add an explicit tie control and fit the tie model rather than forcing noise into binary outcomes.

## 6. Uncertainty, diversity, and discovery

For a pair probability near 0.5, predictive entropy is high, but that can mean either **aleatoric uncertainty** (the user genuinely sees a near-tie) or **epistemic uncertainty** (the model lacks evidence). Only the latter is reliably reduced by more labels.

For the linear head, uncertainty need not require a large neural network. Practical choices are Bayesian logistic regression, a Laplace approximation, or 5–10 L2-regularized heads trained on bootstrap resamples and different seeds. Deep ensembles are a well-supported general uncertainty baseline ([Lakshminarayanan, Pritzel & Blundell](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)), but here the ensemble members should remain small and cheap.

For unseen candidate \(x\), maintain ensemble mean \(\mu(x)\) and standard deviation \(\sigma(x)\). A useful acquisition score is an upper-confidence rule

\[
a(x)=\mu(x)+\beta\sigma(x)+\gamma q(x),
\]

where \(q(x)\) is a separate, bounded generic-quality prior. The queue then applies a diversity penalty such as maximal marginal relevance, subtracting similarity to already-selected candidates. Alternatives such as Thompson sampling are attractive because they naturally vary exploration without assigning every uncertain image an unrealistically high permanent score.

Uncertainty must be paired with out-of-distribution checks. A candidate far from the labeled embedding support can receive an extreme score for the wrong reason. Record nearest-neighbor distance and source/category novelty, cap model confidence outside supported regions, and route some of those items through the explicit exploration quota.

The discovery metric is not “predicted Elo.” Unseen images have no Elo. The model proposes high-utility candidates; the user then compares them, and only judged images enter the collection ranking.

## 7. Training schedule and promotion gates

Training after 20 choices is useful as an end-to-end smoke test, not evidence that the model is ready to drive discovery. Model-guided acquisition should begin only after the comparison graph is connected enough and a frozen-head baseline beats chance on held-out data with useful calibration.

### Staged capacity

1. **Baseline:** frozen OpenCLIP ViT-B/32, normalized embedding, L2-regularized zero-bias linear Bradley–Terry head.
2. **Encoder bake-off:** on a fixed split, compare OpenCLIP, DINOv2, and SigLIP/SigLIP 2 with identical head capacity and tuning budget.
3. **Head bake-off:** after about 500–1,000 diverse labels, compare the linear head with a very small two-layer head, interaction features, and the uncertainty ensemble.
4. **Representation adaptation:** only after several thousand diverse comparisons, test last-block tuning or a parameter-efficient adapter such as [LoRA](https://openreview.net/forum?id=nZeVKeeFYf9). Do not jump directly to full encoder fine-tuning.

Label count alone is not a gate: thousands of repetitive comparisons from one visual cluster have low effective sample size. Promote a more complex model only when all of the following hold:

- lower image-disjoint validation log loss with a bootstrap confidence interval that excludes no improvement;
- no material regression in source/photographer holdouts, calibration, or hidden-repeat behavior;
- gains persist across at least three data splits or temporal folds and random seeds;
- hyperparameters were chosen without touching the final test set;
- hosted CPU embedding/training latency, transfer, and artifact size stay within the product budget.

Every artifact should store the comparison cutoff, split IDs, encoder and weight hash, preprocessing manifest, random seed, hyperparameters, source commit, and metrics. Retraining should create immutable versioned artifacts and switch the active model only after gates pass, so rollback is exact.

## 8. Acquisition and crawler design

The crawler is a staged acquisition system, not a general-purpose web scraper:

```text
provider API -> rights/provenance gate -> download/decode -> technical-quality gate
             -> exact/perceptual dedupe -> embedding + weak quality prior
             -> taste UCB -> diversity/source quotas -> comparison queue
             -> user choices -> retraining -> improved acquisition
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

The scheduler should maintain explicit budgets for exploitation, uncertainty, random exploration, new sources, underrepresented embedding clusters, and underrepresented creators. Without those budgets, a taste model trained on its own selected data creates a feedback loop: it sees more of what it already understands, becomes more confident there, and mistakes narrowness for taste.

## 9. Evaluation and leakage control

### Three questions require three splits

1. **Future choices on known images:** a rolling chronological comparison holdout may contain images seen earlier. This measures online stability and changing judgment, but not generalization to new photographs.
2. **Taste prediction for new images:** an image-disjoint split must keep every image—and every comparison incident to it—out of training. This requires planning evaluation pools prospectively; randomly splitting edges is not image-disjoint.
3. **Discovery beyond familiar sources:** source-, photographer-, and near-duplicate-cluster holdouts test whether the model learned visual taste rather than source identity or a photographer signature.

Use validation folds for encoder, regularization, acquisition, and calibration choices, then evaluate once on the locked test set. Group bootstrap resampling by image cluster or labeling session rather than pretending comparison edges are independent.

### Metrics

- **Primary predictive metrics:** pairwise log loss and accuracy on user labels.
- **Probability quality:** Brier score, reliability diagram, and expected calibration error.
- **Ranking quality:** Kendall or Spearman correlation against a sufficiently compared held-out item ranking, plus top-k regret or NDCG when the held-out graph supports it.
- **Label efficiency:** learning curves versus number of comparisons, with random pairing and repeated sorting as baselines.
- **Discovery yield:** fraction of proposed candidates that reach the collection's top decile after a minimum comparison count, median eventual percentile, labels per accepted top item, creator/source novelty, and embedding diversity.
- **Human consistency:** hidden-repeat agreement, reversed-side agreement, skip rate, session-duration effects, and left/right bias. This is a noise ceiling, not a model score.

### Common leakage paths

- the same photograph as a thumbnail, crop, recompression, monochrome conversion, or mirror across train and test;
- a burst or near-identical series from one photographer across splits;
- repeated comparisons of the same pair with only side order changed;
- provider awards, popularity, favorites, captions, or photographer names passed into the taste head;
- tuning encoder, preprocessing, or thresholds after inspecting the locked test set;
- evaluating only candidates selected by the current model, which hides misses outside its feedback loop.

Deduplicate before splitting, keep metadata out of the personal visual head, maintain a random audit stream, and record all split logic in the private model manifest.

## 10. Chosen v1 design and tradeoffs

| Decision | v1 choice | Reason | Revisit when |
|---|---|---|---|
| Personal objective | Feature-aware logistic Bradley–Terry | Directly matches pairwise labels and generalizes through image features. | Stable residual cycles or category-conditioned failures appear. |
| Encoder | Frozen OpenCLIP ViT-B/32 | Already integrated, CPU-feasible, reproducible, and a credible semantic baseline. | The controlled DINOv2/SigLIP bake-off has enough disjoint labels. |
| Collection rank | Live Elo with immutable comparison history | Immediate feedback matches the product requirement; retained labels make a later item-only BT refit reproducible. | Add periodic BT only after graph coverage supports it; a Bayesian ranker may later add intervals. |
| Cold start | Diverse editorial/open-access seeds and repeated sorting | Coverage matters more than premature model uncertainty. | Held-out performance and calibration justify active model control. |
| Active queries | Entropy + ensemble disagreement + graph repair + exploration, then batch diversity | Balances learnability, connectivity, and novelty. | Logged ablations show a better allocation. |
| Generic aesthetics | Soft intake prior only | Removes obvious low-value tail without overwriting personal taste. | Its incremental discovery yield is proven on random audits. |
| Fine-tuning | Frozen encoder | Hundreds of labels cannot safely support millions of adapted weights. | Several thousand diverse labels and all promotion gates pass. |
| Crawling | Official, rights-explicit provider adapters | Reproducible metadata, controllable rate limits, and auditable licenses. | A new provider offers an equally explicit compatible API. |

## 11. Rights-aware sourcing

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

## 12. Primary-source reading list

### Preference and ranking

- Thurstone, [*A Law of Comparative Judgment*](https://doi.org/10.1037/h0070288) (1927).
- Bradley & Terry, [*Rank Analysis of Incomplete Block Designs: The Method of Paired Comparisons*](https://academic.oup.com/biomet/article-abstract/39/3-4/324/326091) (1952).
- Davidson, [*On Extending the Bradley–Terry Model to Accommodate Ties*](https://doi.org/10.1080/01621459.1970.10481082) (1970).
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
