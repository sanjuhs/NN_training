# Tiny Transformer Conference FAQ

Date:

- `2026-03-13`

Purpose:

- help defend the current transformer thesis at a research conference
- answer both technical and business questions
- provide a short plain-language answer and a deeper technical answer for each topic

Scope of this FAQ:

- current project: `V2A-over-training-old-nn`
- current model: tiny transformer encoder for `audio -> 52 blendshapes + 7 pose`
- current best checkpoint: `tiny_transformer_full10s_l1_best.pth`

---

## 1. What is the one-sentence thesis?

### Plain Answer

We built a very small transformer that converts speech features into face animation controls, and we proved it can learn the mapping on a speaker-specific dataset with real `10-second` context while remaining small enough for browser deployment through ONNX.

### Technical Answer

The core thesis is that an intentionally small transformer encoder can replace or supplement the old TCN-style audio-to-blendshape mapping in a low-latency animation pipeline. The key contribution is not frontier-scale modeling. The contribution is that a compact sequence model with only `432,443` parameters can consume `80`-dimensional framewise log-mel features, model up to `10 seconds` of context at `100 Hz`, and produce `59` synchronized animation outputs per frame in a format that remains compatible with ONNX and browser inference.

---

## 2. What problem are you solving?

### Plain Answer

We want speech-driven facial animation from audio alone, with a small model that can run cheaply and eventually in real time.

### Technical Answer

The task is sequence-to-sequence regression:

- input: per-frame speech features
- output: per-frame facial animation controls

More concretely:

- input tensor: `(batch, sequence_length, 80)`
- output tensor: `(batch, sequence_length, 59)`

The `59` outputs are:

- `52` ARKit-style blendshape values
- `7` pose-related values

This is useful for digital humans, avatar animation, lightweight content creation, dubbing pipelines, low-cost telepresence, and local inference animation systems.

---

## 3. Why does this matter from a business perspective?

### Plain Answer

Because animation is expensive, manual facial keyframing is slow, and small speech-driven animation models can drastically reduce cost for many content pipelines.

### Technical Answer

The business case is not just academic curiosity. It is cost compression and workflow simplification.

Business value drivers:

- reduce animator time for talking-head content
- generate first-pass facial animation automatically
- support low-cost avatar systems
- allow client-side or edge inference instead of costly cloud inference
- enable faster iteration for games, VTubing, education, social media, and localization

The important commercial angle is the small-model constraint. If the model were huge, the cost and deployment friction would undermine many of these use cases. The thesis is stronger because the model is intentionally tiny.

---

## 4. Why use a transformer here at all?

### Plain Answer

Because speech-to-face is a time-sequence problem, and transformers are good at learning relationships across time.

### Technical Answer

Speech animation is temporally dependent. A current mouth or facial frame depends not only on the instant acoustic snapshot but also on nearby phonetic and prosodic context. A transformer encoder is attractive because self-attention can model dependencies over a wide temporal window without forcing everything through a fixed convolutional receptive field.

Why it was appropriate here:

- we wanted `10-second` context
- we wanted a clean sequence model
- we wanted something exportable to ONNX
- we wanted a small architecture that is easy to reason about

This was not chosen because “transformers are fashionable.” It was chosen because wide-context sequence modeling is the central requirement.

---

## 5. Why not keep the old TCN?

### Plain Answer

The old TCN path was the previous attempt, but it was not behaving well enough, and we wanted a cleaner small-context-plus-long-context baseline.

### Technical Answer

A TCN is a legitimate baseline, but it has tradeoffs:

- receptive field is architecture-dependent
- long context often needs deeper or more dilated stacks
- debugging the effective context can be less intuitive
- browser export and practical experimentation were easier to standardize around the new transformer path

This project phase was explicitly about overtraining and proving the pipeline. The transformer path gave a clearer route to:

- explicit `10-second` windows
- clean ONNX export
- architectural simplicity
- easier experimentation with a modern sequence model

This does not mean TCNs are universally worse. It means the transformer was a better fit for this phase and this constraint set.

---

## 6. Why only 3 layers?

### Plain Answer

Because we wanted the smallest model that still works, not the biggest model that sounds impressive.

### Technical Answer

The chosen default was:

- `3` encoder layers
- `d_model = 128`
- `4` heads
- feedforward size `256`

This is a deliberate efficiency point. More layers might improve capacity, but they also increase:

- training time
- inference cost
- export size
- deployment friction
- overfitting risk under limited data

At this stage, the design target was not maximal benchmark score. It was a compact, working model that can be defended as proportionate to the task and budget.

---

## 7. Why `d_model = 128` and not something larger?

### Plain Answer

Because `128` is big enough to learn useful temporal patterns but still small enough to be cheap and deployable.

### Technical Answer

The model’s representational width has to balance:

- enough capacity to capture phonetic and prosodic structure
- low enough parameter count for fast iteration
- manageable ONNX size
- feasible browser inference

For this project, `d_model = 128` is a reasonable compromise. It keeps the total model size around `1.65 MB`, which is unusually lightweight for a transformer-based sequence model.

---

## 8. Why 4 attention heads?

### Plain Answer

Enough to let the model look at time in different ways, but not so many that it becomes wasteful.

### Technical Answer

Multi-head attention allows different learned subspaces to attend to different temporal patterns. With `d_model = 128`, `4` heads gives a clean `32` dimensions per head. This is a balanced design:

- enough head diversity to separate different timing cues
- low compute overhead
- stable dimensionality for a tiny model

Using many more heads would be unnecessary for this model size and dataset regime.

---

## 9. What exact architecture are you using?

### Plain Answer

A tiny transformer encoder with positional encoding and a small output head.

### Technical Answer

Current architecture:

1. input projection from `80 -> 128`
2. sinusoidal positional encoding
3. `3` transformer encoder layers
4. normalized output head back to `59` values
5. output range shaping:
   - sigmoid on blendshape channels
   - `tanh * 0.2` on pose channels

Current model metadata:

- parameters: `432,443`
- architecture name: `tiny_transformer_encoder`
- max configured sequence length: `1200`

---

## 10. What are the “tokens” here?

### Plain Answer

Each time frame is effectively one token.

### Technical Answer

This is not a text transformer. The tokenization is temporal:

- one frame of log-mel features corresponds to one time-step token
- each token has `80` feature values

At `100 Hz`, `10 seconds` means:

- `1000` time steps
- therefore `1000` effective sequence tokens

So the context window is best thought of as `1000 audio frames`, not text tokens.

---

## 11. How much context does the model have?

### Plain Answer

It is trained on `10-second` windows, which means about `1000` frames of temporal context at `100 Hz`.

### Technical Answer

The rebuilt full-data training set uses:

- `sequence_length_frames = 1000`
- frame rate `100 Hz`

That gives `10 seconds` of per-window context. For longer audio, inference is chunked into overlapping `10-second` windows and blended back together, so the production pipeline supports arbitrary clip lengths without requiring the model to process an unbounded sequence in one pass.

---

## 12. What exactly is the output?

### Plain Answer

For every frame, the model predicts `59` numbers that describe face shape and head motion.

### Technical Answer

Per frame output:

- blendshape channels `0:52`
- pose channels `52:59`

Output tensor shape:

- `(batch, sequence_length, 59)`

Fixed `10-second` inference gives:

- `1000 x 59` at `100 Hz`
- `300 x 59` after export to `30 FPS`

Arbitrary-length inference gives:

- approximately `round(T * 30)` frames for an input of `T` seconds

---

## 13. Why `52 + 7` outputs instead of only `52` blendshapes?

### Plain Answer

Because just the face shape is not always enough. Small head motion improves expressiveness.

### Technical Answer

The dataset targets are not only mouth and facial blendshape values. They also include pose-related components. Keeping those extra `7` outputs preserves compatibility with the existing pipeline and better captures full expressive motion rather than just local mouth articulation.

---

## 14. What loss are you using?

### Plain Answer

Mainly `L1 loss`, which simply means “make the prediction as close as possible to the real value.”

### Technical Answer

Two losses were tested:

1. `L1 only`
2. `L1 + temporal smoothness`

`L1` measures absolute error at each output dimension and frame. Temporal loss applies `L1` to frame-to-frame deltas so the predicted motion change matches the target motion change.

This is appropriate because the task is continuous-valued regression, not classification.

---

## 15. Why did L1 beat the temporal version?

### Plain Answer

Because the temporal penalty made the model slightly smoother, but not more accurate on validation.

### Technical Answer

On the corrected full-data run:

- best `L1` validation loss: `0.07287662368147604`
- best `L1 + temporal` validation loss: `0.07439237220152732`

The temporal term can regularize motion changes, but it also slightly constrains raw fitting. For this dataset and model size, plain `L1` gave the best held-out score.

---

## 16. Is the model trained well?

### Plain Answer

It is trained well enough to defend as a successful narrow result, but not enough to claim production-grade universal generalization.

### Technical Answer

Best corrected full-data model:

- best epoch: `14`
- train loss at best epoch: `0.06325812685198685`
- val loss at best epoch: `0.07287662368147604`
- direct train MAE: `0.06297550350427628`
- direct val MAE: `0.07299423962831497`

Interpretation:

- the model is learning meaningful structure
- the train/val gap is real but not catastrophic
- it is stronger than the temporal variant
- it is appropriate to present as a working prototype or phase result

What would be unsafe to claim:

- fully solved general speech-driven animation
- strong cross-speaker robustness
- production-level naturalness across arbitrary domains

---

## 17. Is this overfit or generalization?

### Plain Answer

Both phases exist. The first tiny dataset was for overfit verification. The later rebuilt dataset is the more serious all-data run.

### Technical Answer

There were two distinct phases:

1. small speaker-specific `10-second` overfit set
2. rebuilt full long-context dataset using about `1.165 hours` of usable data

The first phase answers:

- does the pipeline work at all?

The second phase answers:

- can the same architecture scale to a larger, properly rebuilt long-context training set?

That distinction is important in a defense. Do not blur these two experiments.

---

## 18. Why not diffusion?

### Plain Answer

Because diffusion would be much heavier, slower, and harder to justify for this stage.

### Technical Answer

A diffusion model is interesting for expressive motion generation, but it was not a good fit for this phase because:

- inference is slower
- export and browser deployment are harder
- the project goal is a very small network
- we first needed a deterministic, compact baseline that proves the mapping pipeline

Diffusion is stronger when:

- modeling multimodal motion uncertainty
- generating richer stochastic facial dynamics
- prioritizing realism over simplicity

This work is a compact regression baseline, not a high-cost generative animation system.

---

## 19. Why not MoE?

### Plain Answer

Because mixture-of-experts is useful when you need scale and specialization, but this project is intentionally tiny.

### Technical Answer

MoE could help in a larger system by routing different speech or expression regimes to specialized experts. But for this project it would add:

- routing complexity
- training instability risk
- export complexity
- more engineering surface area

MoE becomes more compelling if the project evolves into:

- multi-speaker training
- emotion-conditioned routing
- multilingual speaking styles
- much larger datasets

For a `432k` parameter demo-oriented model, MoE would be overengineering.

---

## 20. Could MoE work later?

### Plain Answer

Yes, but only if the dataset and product scope become much bigger.

### Technical Answer

A future MoE-style extension could route based on:

- speaker identity
- phonetic regime
- emotional prosody
- language or accent
- voiced versus unvoiced segments

That would be more justified if the system moves from a compact demo to a scalable multi-domain product.

---

## 21. Why not a larger transformer?

### Plain Answer

Because bigger is not automatically better when your goal is cost efficiency and deployment simplicity.

### Technical Answer

Large models add:

- more memory pressure
- more latency
- longer training
- harder browser deployment
- stronger temptation to hide poor data quality behind brute-force scale

The current research question is whether a small transformer is enough to capture the signal. A large model would answer a different question.

---

## 22. What is the main novelty here?

### Plain Answer

The novelty is not inventing a brand-new transformer block. It is showing a practical tiny-transformer replacement for this voice-to-animation pipeline under small-model constraints.

### Technical Answer

The contribution is systems-oriented rather than mathematically novel:

- replacing the older TCN path with a compact transformer
- rebuilding the old short-window dataset into real long-context windows
- preserving compatibility with ONNX and browser inference
- supporting arbitrary clip lengths via chunked inference

This is a pragmatic research contribution centered on deployment-aware sequence modeling.

---

## 23. Why rebuild the dataset?

### Plain Answer

Because the old full dataset was not really `10-second` context. It was thousands of tiny overlapping snippets.

### Technical Answer

The original full artifact was:

- `40743 x 23 x 80`

That is only about `23` frames per window. If you claimed that as a real long-context transformer experiment, the claim would be weak. So the windows were stitched back into continuous sequences and then recut into:

- `798 x 1000 x 80`

This makes the long-context claim honest.

---

## 24. How do you know the rebuilt dataset is correct?

### Plain Answer

Because adjacent windows in the old dataset overlapped, and we verified those overlaps line up exactly except at real segment boundaries.

### Technical Answer

The short-window dataset was generated with:

- sequence length `23`
- step size `11`

That means adjacent windows share `12` frames. We compared overlapping regions between neighboring windows and used mismatch points to detect segment boundaries. Continuous regions were then reconstructed by appending only the non-overlapping tail from each next window.

That reconstruction error was caught and fixed during development. The final corrected rebuilt dataset is the one used for the full-data results.

---

## 25. Why use log-mel features?

### Plain Answer

Because they are a simple, compact, and proven speech representation.

### Technical Answer

Log-mel features are a strong baseline for this problem:

- compact
- robust
- well-understood
- fast to compute
- ONNX/browser friendly

They capture spectral structure relevant to articulation without needing a massive front-end model. A future system could replace them with learned speech embeddings, but log-mels are the right choice for a compact baseline.

---

## 26. Why not use a pretrained speech model?

### Plain Answer

That is a valid future direction, but it would increase complexity and move the work away from the “small deployable model” goal.

### Technical Answer

A pretrained speech encoder like wav2vec-style or HuBERT-style features could improve semantic and prosodic richness. But it would also:

- increase the dependency footprint
- complicate deployment
- increase inference cost
- partially defeat the tiny-model constraint

This project intentionally prioritizes compactness and control over transfer-learning maximalism.

---

## 27. Why not use raw waveform input?

### Plain Answer

Because raw-waveform models are usually more expensive and harder to stabilize for a project like this.

### Technical Answer

Raw waveform modeling pushes more work into the network. For this project that would likely require:

- a larger front-end
- more compute
- more training data
- more tuning effort

Log-mels already expose useful phonetic structure in a compact form, which is more aligned with the current engineering objective.

---

## 28. What business objections should you expect?

### Plain Answer

People may ask whether this is just a cool demo or a product. You should answer that it is an enabling layer for cheaper animation workflows, not the entire product by itself.

### Technical Answer

Likely business objections:

- “Is this defensible or easy to copy?”
- “How does this reduce cost?”
- “Is the market big enough?”
- “Why not just use manual animation or existing facial capture?”
- “Can this work with enterprise privacy constraints?”

Good framing:

- the model is not the whole moat
- the moat can come from pipeline quality, creator tooling, data, deployment, UX, and real-time integration
- the economic value is reduced manual effort and scalable first-pass animation

---

## 29. What is the moat?

### Plain Answer

The moat is not only the neural net. It is the data pipeline, deployment simplicity, quality tuning, and workflow integration.

### Technical Answer

Model weights alone are rarely a durable moat. Real defensibility can come from:

- proprietary aligned audio-animation data
- robust production preprocessing
- inference tooling
- browser or edge deployment
- editing and correction UX
- integration into creator pipelines

The right answer is not “our transformer block is unique.” The right answer is “our end-to-end system is becoming operationally valuable.”

---

## 30. Is this really an LLM problem?

### Plain Answer

No. It is a sequence modeling problem that shares transformer ideas with LLMs, but it is not a language model in the usual sense.

### Technical Answer

This work borrows transformer architecture ideas from the same family as LLMs, but it is not next-token text generation. It is a dense regression model over audio time steps. At an LLM conference, it is useful to say:

- architecturally related
- application-wise very different
- same transformer family, different modality and objective

That keeps the claim honest.

---

## 31. What are the main limitations today?

### Plain Answer

It is still a narrow model and not yet a fully general face animation engine.

### Technical Answer

Current limitations:

- trained on a limited speaker/data regime
- no explicit emotion control input
- no text or phoneme supervision
- no visual discriminator or realism prior
- browser path still depends on ONNX CPU-style execution
- qualitative naturalness is not fully benchmarked yet

This is a strong prototype result, not the final system.

---

## 32. What would you do next if you had more time?

### Plain Answer

Train on more speakers, improve evaluation, and add better conditioning.

### Technical Answer

Most valuable next steps:

1. multi-speaker dataset expansion
2. speaker embedding conditioning
3. explicit emotion or prosody conditioning
4. compare against a better TCN baseline on the same rebuilt dataset
5. add better metrics beyond MAE
6. visual evaluation and perceptual scoring
7. test WebGPU or better browser backends

---

## 33. How would you answer “why not just use a giant multimodal model?”

### Plain Answer

Because that would solve a different problem with much higher cost and much lower deployability.

### Technical Answer

A giant multimodal model may outperform this system in absolute quality, but:

- compute cost would rise sharply
- latency would rise
- browser deployment would likely break
- product economics would worsen

This project is intentionally on the opposite side of the design space:

- tiny
- cheap
- understandable
- exportable

---

## 34. How do you defend using ONNX and browser inference?

### Plain Answer

Because deployment matters. A small model that runs where users actually need it is more useful than a larger model that only works in a lab.

### Technical Answer

The deployment story is part of the research value:

- ONNX export worked
- ONNX matched PyTorch numerically
- the browser demo consumes the same model contract
- long-form inference is chunked and stitched

This is a practical contribution. It shows the architecture is not only trainable but also portable.

---

## 35. What should you say if someone asks whether it is “real time”?

### Plain Answer

Today it is a small near-real-time-friendly model, but exact real-time guarantees depend on the deployment backend.

### Technical Answer

Be precise:

- the model is small enough that real-time deployment is a realistic goal
- browser inference still depends on backend and device
- the current work proves compactness and compatibility, not a fully benchmarked production latency SLA

If asked for honesty, say:

- “I can defend that the model is lightweight and deployment-aware”
- “I cannot yet defend a universal real-time guarantee across all browsers and devices”

---

## 36. How do you defend this work as a thesis and not just engineering?

### Plain Answer

Because the thesis is about the design tradeoff: how small a modern sequence model can be while still learning useful speech-to-animation mappings and remaining deployable.

### Technical Answer

This is a defensible thesis if framed correctly:

- not “I invented transformers”
- but “I explored a compact transformer design for speech-driven facial animation under deployment and cost constraints”

The academically defensible elements are:

- model choice tradeoffs
- dataset reconstruction methodology
- evaluation of compact sequence architectures
- deployment-aware design
- alternative comparisons and limitations

---

## 37. What are the best “defense lines” if someone attacks the work?

### Plain Answer

Stay honest, stay narrow, and keep repeating that this is a compact, deployment-aware prototype with verified long-context training and working ONNX/browser integration.

### Technical Answer

Strong defense points:

- “This is not claiming universal facial animation.”
- “The contribution is a tiny long-context transformer that actually deploys.”
- “We rebuilt the dataset properly instead of pretending `23-frame` windows were long context.”
- “We compared `L1` and `L1 + temporal` and chose the empirically stronger result.”
- “We verified ONNX parity and arbitrary-length inference.”

Weak defense points to avoid:

- claiming state of the art
- claiming production quality everywhere
- claiming transformer superiority in the abstract

---

## 38. What should the audience remember?

### Plain Answer

That a very small transformer can drive a face animation pipeline from audio with real `10-second` context and browser-compatible deployment.

### Technical Answer

The key memory hook is:

- tiny transformer
- rebuilt full-context data
- `10-second` temporal modeling
- `59` animation outputs
- ONNX/browser portability
- arbitrary-length inference through chunked stitching

That combination is the actual contribution.
