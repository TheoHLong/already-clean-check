# Preregistration: Stage 0, InceptionV1 mixed4e:55

Commit this file and `prereg/4e55.json` **before** running the analysis cells. The git
timestamp is the evidence that the predictions came first. Changes after seeing data go in
a new section below, with a reason.

## Claim under test

The essay says the canonical "cat faces + car fronts" polysemantic neuron 4e:55 is a single
V shape / pointed front detector, and that round shapes (an O) drive its negative pole.

## Predictions (fill in, then commit)

| ID | Prediction | Measure | Supports the claim if | Contradicts the claim if |
|----|------------|---------|------------------------|---------------------------|
| P1 | Sanity check, not a prediction (already seen in the pilot): the essay's circuit connections exist | weight rank of 4d:493/477/201 (inputs), 5a:94/352/317 (readers) | pilot: ranks 1, 8, 15 and 1, 1, 15 | n/a |
| P2 | New pointed-front classes I chose (`predicted_positive`) drive it | class mean of spatial max, AUC vs all other classes | AUC > 0.75 | AUC < 0.6 |
| P3 | New round classes I chose (`predicted_negative`) push it negative | class mean of spatial min, AUC vs all other classes | AUC > 0.75 | AUC < 0.6 |
| P4 | Synthetic pointed shapes beat round ones | mean normalized response: V, triangle vs O, disk | pointed > round in both polarities | round >= pointed |
| P5 | The top 100 images are many classes sharing one shape, not two unrelated clusters | class composition + visual inspection of crops | >= 10 classes, crops all show a V-like region | two clean clusters (cats, cars) with no shared shape |

Notes for P4: synthetic line drawings are out of distribution for a layer this deep, so
responses are normalized by the unit's 99th percentile on natural images. A pilot run in
the sandbox (before any natural images) already showed U shapes responding as strongly as
V shapes; P4 should be read with that in mind, and a "converging arms vs closed contour"
alternative should be considered alongside "pointed vs round".

## Analysis plan

1. Cache pre-ReLU spatial max/min of mixed4d, mixed4e, mixed5a on all 50k val images
   (Resize 256, CenterCrop 224).
2. Run P1 to P5 exactly as coded in `acc/stage0.py`, no changes to class lists after commit.
3. Report every prediction, including failures.

## Deviations (after seeing data)

_None yet._
