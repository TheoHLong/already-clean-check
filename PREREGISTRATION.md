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

---

# Preregistration: Stage 0b, Zoom In's "superposition" car circuit

Written and committed before any mixed4b/4c activation, flip response or attribution was
computed. Machine readable version: `prereg/zoomin_car.json`. Code: `acc/zoomin.py`.

## Claims under test

* **Zoom In (2020):** a car detector in mixed4c spreads its feature into several dog detecting
  neurons in mixed4d; the model stores the car there "in superposition" to read it out later.
* **Goodfire essay (2026):** those neurons are generalized pose neurons (cars and dogs that share a
  pose); nothing downstream reads the combination back out; feature visualization made them look
  like pure dog neurons.

## Selection rules (fixed now)

* `C`: the mixed4c channel with the highest share of the 12 car classes among its top 100 images.
* `T`: the 8 mixed4d channels with the largest positive weight from `C` (exact 1x1 weights, expanded
  weights for 3x3/5x5 branches).
* `D`: members of `T` whose top 100 images are at least half dogs.

## Predictions

| ID | What | Supports essay | Supports superposition / contradicts essay |
|----|------|----------------|---------------------------------------------|
| P1 | circuit exists (descriptive) | car share of C >= 0.5 and \|D\| >= 3 | n/a |
| P2 | cars among D's top 100 images | median car share >= 0.05 (base rate 0.012) | near base rate |
| P3 | shared pose, via a label free mirror test: each channel's preference for the photo vs its mirror image, measured separately on dog and car images | same sign in >= 75% of laterally tuned D channels and >= 20 points above other mixed4d channels | <= 60% |
| P4 | readout: share of positive car attribution into car selective mixed4e channels flowing through D | < 0.05 | >= 0.10 |
| P5 | the 5 mixed4e channels that most strongly combine D | all car share < 0.25 | any car share >= 0.5 |

Known limits: the mirror test only sees left/right pose, not frontal vs side; attribution is
gradient x activation on each target's 50 strongest car images; car and dog classes are fixed lists.
