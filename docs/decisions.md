# Decision log

Research code accumulates choices that make no sense six months later unless someone wrote down why. This file records the ones that shaped the method, where each came from, and whether it is settled. When you change one of these, edit the entry rather than deleting it, and say why the old choice was wrong.

Provenance tags:

- **Inherited** means it comes from the GABRIL paper and we did not change it.
- **Our call** means we chose it for this project.
- **Open** means it was a placeholder or a guess and should be revisited.

## Encoder: factorized ViViT

**Our call.** The paper regularizes the final convolutional feature map of a CNN. We needed a transformer analogue, and we picked a factorized ViViT: a spatial transformer per frame followed by a temporal transformer over the per-frame CLS tokens. The reason is that this design gives clean per-frame spatial attention maps, which is the direct counterpart of a per-frame saliency map. A joint space-time transformer would mix attention across frames and make "where did the model look in this frame" harder to read. Grayscale 84x84 Atari frames are small enough that the factorized model is cheap to train.

Status: settled for these experiments. The paper's own follow-up work, AutoFocus-IL, is worth reading if you do not want to collect gaze at all.

## Gaze loss form: KL divergence with gaze as the target

**Our call.** The paper uses an L2 loss between a learned gaze predictor and the gaze mask. We instead treat the spatial attention over patch tokens as a probability distribution and minimize `D_KL(gaze || attention)`. Attention comes out of a softmax, so it is already a distribution, and a KL pulls the whole distribution toward gaze in one shot instead of training a separate predictor.

The direction matters and is easy to get backwards. Putting gaze first means gaze is the target distribution and the model is penalized for placing attention where gaze is low. The reverse direction, `D_KL(attention || gaze)`, would force attention to be near zero everywhere gaze is near zero, which is a much harsher constraint and not what we want.

Status: settled. The exact reduction is not, see below.

## Which token the gaze loss reads

**Open, and probably wrong.** In `AuxGazeFactorizedViViT` the extracted attention is token index 0 (`vivit.py:595`), which is the policy token. The dedicated gaze token is index 1 and is never read by the loss or fed to the classifier. Either we meant to read the gaze token, or the gaze token is pointless and the plain `FactorizedViViT` is the honest model. Resolve this with Yutai before drawing conclusions from the auxiliary model.

## Auxiliary gaze token

**Our call.** `AuxGazeFactorizedViViT` adds a second spatial token, a policy token and a gaze token, so the gaze loss can shape a token that the classifier does not depend on. The intent was to keep the gaze regularization from distorting the policy representation. Given the token bug above, this intent is not currently realized.

Status: open.

## Reduction order: mean_then_kl vs kl_then_mean

**Open.** `mean_then_kl` averages the attention heads first and computes one KL per layer, batch item, and frame. `kl_then_mean` computes a KL per head and averages after. The config default is `mean_then_kl` (`config.py:185`), but the two main Slurm sweeps pass `--gaze-loss-mode kl_then_mean` explicitly (`slurm/train-carc.slurm:30` and the Snoopy script), so the runs that produced the results use `kl_then_mean`. This mismatch between the default and what was actually run is a trap. Pick one, make the default match it, and document the choice.

## Layer selection: last by default

**Our call, supported by the ablation.** `--gaze-loss-layers` defaults to `last`. The intuition is that the last spatial layer is closest to the decision, so shaping its attention should matter most. The Breakout ablation backs this up hard. Putting the gaze loss on the first layer gives a zero return for every seed in both architectures, while last-layer runs score in the double digits. The `all` setting sits in between.

Status: settled on `last` unless you have a reason.

## Head selection: all, or the first half

**Open.** `--gaze-loss-heads` defaults to `all`. The `half` option selects the first `max(1, spatial_heads // 2)` heads by index (`train.py:167`), not by attention magnitude or any learned criterion. On Breakout, `half` sometimes beat `all` and sometimes did not. If you pursue this, select heads by a meaningful signal rather than by index.

## Gaze weight: lambda = 0.5

**Open.** We set `--lambda-gaze` to `0.5` and never swept it. In the web-agent project the same weight is `0.001`, because there the base model is already strong and a large weight would wreck it. The right value here is unclear and worth a sweep. If the gaze loss seems to hurt, this is the first knob to question.

## Gaze heatmap construction

**Inherited, with an open parameter.** The per-frame gaze window is 41 steps (`short_memory_length=20`, `stride=2`, `num_sigmas=41`, `dataset.py:536`). Points are spread into a soft Gaussian with `gaze_sigma=15`, the window decays over time with `gaze_alpha=0.7`, and the blur grows with `gaze_beta=0.99`. The window size is the one number nobody has confirmed. The web-agent project uses roughly 200 ms, which is a different formulation of the same question. Ask Yutai what GABRIL used and whether the window should include future frames.

Status: inherited from the paper's preprocessing; window size open.

## Frames and patches

**Our call.** `--frame-stack 4` and `--frame-skip 4`, so the model sees four frames 4 apart, which is the standard Atari convention and matches the dataset's `_fs4_human` files. `--patch-size 6` on 84x84 frames gives a 14x14 grid, 196 patch tokens per frame. Small enough to train, large enough to localize gaze.

Status: settled.

## Class weighting

**Inherited.** The BC loss uses inverse-frequency class weights, square-rooted and clamped to `[1, 10]` (`train.py:292`). Atari action distributions are heavily skewed, and the clamp keeps rare actions from dominating. On by default.

Status: settled.

## Determinism

**Our call.** Seeds are fixed, TF32 is off, cuDNN runs deterministic, and the DataLoader generators are seeded and saved in the checkpoint so a resumed run shuffles the same way (`train.py:731`, `checkpoint.py`). This makes runs comparable, which matters because the variance across seeds is large enough that a bad seed can flip a conclusion.

Status: settled. Do not turn it off for a final number.

## Train and validation split

**Open, known weak.** `--train-pct 0.95` cuts the concatenated data sequentially, first 95 percent train, last 5 percent validation (`train.py:377`). It is not shuffled and not grouped by episode, so the validation set is whatever happened to be at the end of the file. Test is a live emulator rollout, which sidesteps this, but do not trust the validation number until the split is fixed.

## Checkpoint every epoch, auto resume

**Our call.** A full checkpoint, including optimizer, scheduler, scaler, and RNG state, is written each epoch, and training resumes automatically if one exists. This is because cluster jobs get cut off at the wall clock and we did not want to lose a 500 epoch run. It costs some disk and makes runs slightly non-comparable if the resume logic ever misfires.

Status: settled.

## Web-agent decisions

These live in the `gaze-web-nav` project. Recorded here because they are the same research question with different constants.

**Soft KL on the last transformer layer, lambda around 0.001.** Our call. UI-TARS-1.5-7B is already instruction-tuned on UI data, so the gaze term is a gentle nudge, not a primary signal. The `0.001` is a starting point, not a tuned value.

**Gaze loss only on spatially grounded actions (click, double click, right click, drag, scroll).** Our call. Typing and hotkeys put the eyes on the keyboard, so the gaze samples are invalid or off-screen and the action has no screen target. This masking matters more than it sounds; including type actions would add noise to the target.

**Gaze window of roughly 200 ms before each action's decision timestamp.** Open. Chosen to match the decision moment. Confirm with Yutai.

**Absolute pixel coordinates via ByteDance `smart_resize`, never 0 to 1000 normalized.** Our call, and a correction. UI-TARS-1.5 is Qwen2.5-VL based and expects absolute pixels. The old preprocess normalized to 0 to 1000, which was wrong. This is issue A in that repo's `HANDOFF.md`.

**Collect gaze while the human performs the actions, never by replaying a script.** Our call. Gaze is only informative because it captures where attention went while deciding. If a script acts and the human watches, gaze follows the cursor instead of leading the action, and the comparison behavior on the tasks that matter never happens.

**Split by task, seed 42, with 15 evaluation task IDs forced into test.** Our call. A task and all of its demonstrations must live in the same split, or the model has effectively seen the test task. The forced IDs are the held-out evaluation set.

**Map gaze into screenshot pixels using the screen geometry saved at recording time.** Our call. The gaze tracker reports fractions of the physical screen, and the screenshot is only the content viewport. The offset is not recoverable after the fact, which is why `session_info.json` is captured once at startup and why the window must not move mid-recording.
