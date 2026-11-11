# Reliability Calibration for CNN and ViT Models

This repository evaluates how calibration affects the confidence scores of
CNNs and vision transformers on CIFAR-10 and CIFAR-100. Each calibrator is
fitted using saved logits and labels from an INT8 model. The fitted transform
is then folded into the model, which is re-exported as INT8 TFLite and
evaluated to measure how well its confidence scores reflect its accuracy.

| Step | Inputs | What it does | Outputs |
| --- | --- | --- | --- |
| 1. Select base models | Model name, CIFAR dataset, saved split, Stage 1 YAML | `hyper_search_train.py` uses Optuna to maximize deployed INT8 accuracy and minimize estimated GR-compressed TFLite size. | Optuna trial results and model exports |
| 2. Tune calibrators | Selected model ID and manifest, saved INT8 logits and labels, split, Stage 2 YAML | `hyper_search_calibration.py` tunes temperature scaling or classwise affine calibration to minimize reliability risk, with a penalty for exceeding the allowed accuracy drop. | Optuna trial scores and the fitting settings used in each trial |
| 3. Evaluate calibrated models | Model checkpoint, baseline TFLite bundle, saved split and calibration inputs, deployment JSON | The CIFAR-10 or CIFAR-100 runner fits the calibrator for each job, folds it into the model, and evaluates the INT8 TFLite export. | Evaluation results, raw TFLite size, fitted parameters, and model exports |

If you already have the trained models and the saved inputs for the recorded
runs, go directly to Step 3.

## Before you start

Run the commands from the root of this calibration repository. The
deployment runners were tested on Linux with Python 3.12.13 and the package
versions in `requirements.txt`.

Stage 1 and Stage 2 use the open-source Optuna library through the shared
`OptunaTuner` in IFX MLPerfTiny. The search scripts also import IFX's data,
training, and TFLite helpers. To run the search commands, activate the IFX
PyTorch environment and set `IFX_MLPERF_TINY_ROOT` to the absolute path of
your IFX MLPerfTiny checkout.

```bash
export PYTHONPATH="$PWD:$IFX_MLPERF_TINY_ROOT/benchmark:$IFX_MLPERF_TINY_ROOT/benchmark/training_torch/image_classification${PYTHONPATH:+:$PYTHONPATH}"
```

The branch includes the Python code, six Optuna study YAMLs in `studies/`,
and three deployment plans in `configs/`. To rerun a recorded experiment,
you also need its saved data split, cached logits, trained model weights,
and baseline TFLite model bundle. Set the variables below to the locations of your saved files. Access to
those experiment files must be arranged separately.

| Path variable | What to supply |
| --- | --- |
| `CIFAR10_ROOT` | The parent folder of the extracted CIFAR-10 data. For example, if the batch files are in `/data/cifar10/cifar-10-batches-py/`, set `CIFAR10_ROOT=/data/cifar10`. |
| `CIFAR100_ROOT` | The parent folder of the extracted CIFAR-100 data. For example, if the `train` and `test` files are in `/data/cifar100/cifar-100-python/`, set `CIFAR100_ROOT=/data/cifar100`. |
| `INPUT_ROOT` | The root containing the saved split and calibration input exports at the paths referenced by the deployment plan. The exports are `.npz` files containing logits and labels. |
| `ARTIFACT_ROOT` | The baseline TFLite model bundle with its `manifest.json`. The runner looks up the selected candidate in this manifest. |
| `WEIGHTS_ROOT` | The folder containing the PyTorch model weights. For CIFAR-10, supply `resnet8.pt` with `--model resnet8` or `vit_tiny_cifar10.pt` with `--model vit_tiny`. For CIFAR-100, pass the checkpoint for the requested model with `--checkpoint`. |
| `OUTPUT_ROOT` | A writable directory outside the checkout for deployment outputs. |
| `SEARCH_OUTPUT_ROOT` | A writable directory outside the checkout for the Optuna runs. |

Keep cached logits, checkpoints, model exports, Optuna journals, and generated
results outside Git.

## 1. Select a base model with Optuna

Each trial trains a candidate model, applies post-training quantization, and
evaluates the INT8 TFLite export on the Stage 1 validation split. The trainer
reuses the shared IFX `OptunaTuner` from `benchmark/util/optuna_search.py`,
the IFX PyTorch training and evaluation helpers, and IFX's
`compressed_size()` helper. Optuna maximizes deployed INT8 accuracy and
minimizes `rg_compressed_size`, the **estimated GR-compressed TFLite size**.

| Model | `--model_name` | Stage 1 study |
| --- | --- | --- |
| ResNet-8 | `resnet8` | `studies/stage1_resnet8_cifar10.yaml` |
| ViT-Tiny | `vit_tiny_cifar10` | `studies/stage1_vit_cifar10.yaml` |
| ResNet-32 | `resnet32_cifar100` | `studies/stage1_resnet32_cifar100_search60.yaml` |
| MobileNetV2 | `mobilenet_v2_cifar100` | `studies/mobilenet_v2_search60.yaml` |

For example, set `CIFAR100_SPLIT` to the saved CIFAR-100 experiment split JSON
and launch the ResNet-32 study:

```bash
python hyper_search_train.py \
  --model_name resnet32_cifar100 \
  --cifar100_path "$CIFAR100_ROOT" --split_path "$CIFAR100_SPLIT" \
  --ssd studies/stage1_resnet32_cifar100_search60.yaml \
  --files_root "$SEARCH_OUTPUT_ROOT/stage1"
```

The study writes Optuna trial results and model exports under the directory
passed to `--files_root`.

For CIFAR-10, pass `--cifar10_path "$CIFAR10_ROOT"` and the corresponding saved
CIFAR-10 split. Use the ViT-Tiny YAML to launch a search with the settings
saved in the repository. The exact settings used for the final ViT-Tiny run
are unknown.

## 2. Tune a calibrator with Optuna

Stage 2 reuses the IFX tuner and keeps the base model fixed. For example,
`resnet8_trial_33` identifies the ResNet-8 model chosen from Optuna trial 33
in Stage 1. The two Stage 2 YAMLs tune how temperature scaling and
classwise affine calibrators are fitted. Each trial fits one calibrator on
saved INT8 logits and scores it on a separate validation subset.

**Optuna objective.** Minimize `objective_reliability_risk`, which weights
negative log-likelihood (NLL), Brier score, expected calibration error (ECE),
and high-confidence error equally. The objective adds a penalty if top-1
accuracy drops by more than 0.5 percentage points relative to the
uncalibrated model on that split.

**Classwise-affine fitting loss.** The calibrator fit uses equal NLL and
Brier weights plus separate regularization for the per-class scales and
biases.

The launch template below expects the saved candidate ID and manifest
(`CANDIDATE_ID` and `CANDIDATE_MANIFEST`), a directory containing its saved
Stage 2 logits (`CACHE_ROOT`), and the random subset budget and seed for the
run being reproduced (`SELECTION_BUDGET` and `SELECTION_SEED`). These are
inputs to Stage 2. The template assumes the candidate bundle and cache have
already been prepared.

Use the same model and cached logits for the two studies. Set the common
arguments once, then run the study you want.

```bash
stage2_args=(
  --dataset cifar100 --cifar100_path "$CIFAR100_ROOT"
  --split_path "$CIFAR100_SPLIT"
  --candidate_id "$CANDIDATE_ID" --candidate_manifest "$CANDIDATE_MANIFEST"
  --cache_root "$CACHE_ROOT" --selection_method random
  --selection_budget "$SELECTION_BUDGET" --selection_seed "$SELECTION_SEED"
)

# Tune temperature scaling.
python hyper_search_calibration.py "${stage2_args[@]}" \
  --ssd studies/stage2_temperature_scaling.yaml \
  --files_root "$SEARCH_OUTPUT_ROOT/stage2/temperature_scaling"

# Tune classwise affine calibration.
python hyper_search_calibration.py "${stage2_args[@]}" \
  --ssd studies/stage2_classwise_affine.yaml \
  --files_root "$SEARCH_OUTPUT_ROOT/stage2/classwise_affine"
```

Each study writes its Optuna trial results under the directory passed to
`--files_root`. Use the trial results to identify the fitting settings that
minimize the Stage 2 objective.

The branch update was checked with CLI and YAML validation and
synthetic-logit fits for both Stage 2 calibrators. Full Optuna trials using
the saved artifacts were not rerun as part of that check.

## 3. Fit, export, and evaluate the calibrated models

### 3.1 Choose the model and phase

A **phase** tells the deployment runner which group of jobs to read from the
JSON plan. Each job pairs a calibration configuration with saved fitting
inputs for a particular model and seed.

- **`development`** evaluates the calibration configurations explored during
  development. For example, MobileNetV2 has nine configurations at seed 42.
- **`repeat`** refits and evaluates the configurations specified in the
  repeat plan. For example, MobileNetV2 runs the selected temperature-anchored
  calibration (TA) configuration and the temperature scaling (TS) baseline
  for each seed from 1 to 10, giving 20 jobs.

Both phases evaluate the exported models on the same saved development
split. The Optuna searches in Steps 1 and 2 have their own commands.

Pass a supplied JSON to `--settings`, a model to `--model`, and a phase to
`--phase`. The same CIFAR-10 JSON serves both models. Each invocation runs
the jobs for the requested model and requires that model's checkpoint and
baseline TFLite export.

| `--model` | JSON file for `--settings` | `development` | `repeat` |
| --- | --- | --- | --- |
| `resnet8` or `vit_tiny` | `configs/cifar10_deployment_resnet8_and_vit_tiny.json` | 9 jobs per model | 20 jobs per model |
| `resnet32` | `configs/resnet32_cifar100.json` | — | 20 jobs |
| `mobilenetv2` | `configs/mobilenetv2_cifar100.json` | 9 jobs at seed 42 | 20 jobs across seeds 1–10 |

The supplied ResNet-32 plan supports `repeat` only. Use the JSONs as
provided. The CIFAR-10 and ResNet-32 JSONs list the jobs explicitly. The
MobileNetV2 JSON supplies the configurations, seeds, and input paths from
which the runner builds its jobs. Both deployment runners fit TS with
`fit_temperature()` and TA with `fit_ta_cgsrh()` in
`calibration/anchored_components.py`.

### 3.2 Validate the inputs and run the jobs

Set the dataset, input, artifact, weights, and output paths from
[Before you start](#before-you-start). The following commands validate the
saved inputs and print the job count. Add `--run` to the command to fit the
calibrators, export the calibrated INT8 TFLite models, and evaluate them.

**ResNet-8 on CIFAR-10.** Put `resnet8.pt` in `WEIGHTS_ROOT`. The baseline
bundle's `manifest.json` must include the ResNet-8 export. With the matching
inputs in place, the following command prints
`PASS: 20 CIFAR-10 resnet8 repeat jobs verified.`

```bash
python -m experiments.deploy_cifar10 \
  --model resnet8 --settings configs/cifar10_deployment_resnet8_and_vit_tiny.json \
  --phase repeat \
  --data-root "$CIFAR10_ROOT" --input-root "$INPUT_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" --weights-root "$WEIGHTS_ROOT" \
  --output-root "$OUTPUT_ROOT"
```

For ViT-Tiny, use `--model vit_tiny` with the same JSON and supply
`vit_tiny_cifar10.pt`, its baseline TFLite export, and its saved calibration
inputs.

**ResNet-32 and MobileNetV2 on CIFAR-100.** Use the JSON and checkpoint for
the requested model.

```bash
python -m experiments.deploy_cifar100 \
  --model resnet32 --settings configs/resnet32_cifar100.json --phase repeat \
  --data-root "$CIFAR100_ROOT" --input-root "$INPUT_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --checkpoint "$WEIGHTS_ROOT/resnet32_cifar100.pt" \
  --output-root "$OUTPUT_ROOT"

python -m experiments.deploy_cifar100 \
  --model mobilenetv2 --settings configs/mobilenetv2_cifar100.json --phase repeat \
  --data-root "$CIFAR100_ROOT" --input-root "$INPUT_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --checkpoint "$WEIGHTS_ROOT/mobilenet_v2_cifar100.pt" \
  --output-root "$OUTPUT_ROOT"
```

The runners compute SHA-256 checksums automatically to verify the saved
split, calibration inputs, and baseline TFLite exports against the
recorded files.

### 3.3 Read the results

After a run with `--run`, open **`result.json`** in the job's output
directory to see the evaluation metrics and the paths to its artifacts.
For the CIFAR-10 example, job directories are under
`$OUTPUT_ROOT/cifar10/repeat/resnet8/`. For ViT-Tiny, look under
`$OUTPUT_ROOT/cifar10/repeat/vit_tiny/`. MobileNetV2 repeat jobs are under
`$OUTPUT_ROOT/mobilenet_v2_cifar100_trial_26/repeat/`.

| File | What it contains |
| --- | --- |
| `result.json` | Evaluation metrics and references to the generated files |
| `fitted_parameters.pt` | The fitted calibration transform and its settings |
| `calibrated_model_int8.tflite` | The calibrated model for INT8 TFLite deployment |
| `calibrated_model_pt2e.pt2` | The quantized PyTorch export |
| `runner_audit.json` | At the run root, checksums of the settings, runner, checkpoint, and model manifest |

The deployment exporter reports **raw TFLite file size** in
`tflite_size_bytes`. Stage 1 model selection uses **estimated GR-compressed
TFLite size** in `rg_compressed_size`.

## Additional paper baselines

The baseline examples below use the original baseline code and saved run
inputs. Supply the model's saved INT8 logits, labels, and the fitting settings
for the calibration subset and seed you want to reproduce.

| Method | How to fit it | Output |
| --- | --- | --- |
| Vector scaling (VS) | Run `python -m reliability_calibration.experiments.ta_cgsrh_v2.run_vector_cifar10 --run` for CIFAR-10 or use `run_vector_pilot` in the same package for the CIFAR-100 pilot. Each script reads its saved run inputs. | Fitted per-class scales and biases |
| Matrix scaling (MS) | Use the baseline-capable `hyper_search_calibration.py` with `--ssd reliability_calibration/studies/stage2_matrix_scaling.yaml` and the candidate, manifest, cache, split, and subset arguments shown in Step 2. | Optuna trial results for selecting a full affine calibration fit |
| Structured matrix scaling (SMS) | Use `stage2_structured_matrix_scaling.yaml` for the baseline Optuna study. For the separate SMS follow-up fit, call `fit_sms(logits, labels, configuration, design)` in `sms_followup_baseline.py` with that run's saved configuration and design. | Optuna trial results or fitted SMS parameters, depending on the entry point |
| Standalone CGSR-H | Use the baseline-capable `hyper_search_calibration.py` with `--ssd reliability_calibration/studies/stage2_confusion_guided_residual_hce.yaml` and the matching Step 2 inputs. | Optuna trial results for selecting a sparse affine calibration fit |

These entry points depend on the original baseline code. The deployment
commands in Step 3 cover the TA and TS jobs in the supplied JSON plans.
