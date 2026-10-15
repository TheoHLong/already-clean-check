# Stage 0 results: InceptionV1 mixed4e:55

Run: 2026-10-15, Colab A100, ImageNet val (50,000 images, HF parquet), original Lucid weights.
Predictions: `prereg/4e55.json` as committed in f97e1cb (pushed before any natural-image result).
Raw outputs: Google Drive `MyDrive/acc_runs/stage0/mixed4e_55/`.

Natural-image scale: 99th percentile of the unit's spatial max (pre-ReLU) = 862.7.

## Preregistered checks

| ID | Result | Verdict |
|----|--------|---------|
| P1 circuit (sanity) | inputs 4d:493 / 477 / 201 rank 1 / 8 / 15 of 528; readers 5a:94 rank 2 of 384 (exact), 5a:352 rank 1 and 5a:317 rank 20 of 448 (expanded) | matches essay |
| P2 new pointed classes drive it | AUC(max) vs all other classes = 0.70 | inconclusive (0.6 to 0.75) |
| P3 new round classes push it negative | AUC(min) vs all other classes = 0.67 | inconclusive |
| essay's own examples | AUC(max) pos vs neg = 1.00; pos vs rest = 0.97; neg more negative than rest = 0.85 | replicate (but selected by the author) |
| P4 synthetic pointed > round | V 0.12, triangle 0.13/0.06 vs O 0.01/0.05, disk 0.02 (fraction of natural p99; dark/light) | technically yes, but U = 0.15 is highest and every shape is <= 15% of natural p99 |
| P5 one shared shape, not two clusters | top 100 images: 26 classes, 4.2 bits; all strongest crops are front-facing heads of cats, foxes, weasels, Pomeranian/papillon with upright pointed ears | supported; cars rare in top 100 (2 beach wagons) |

Per class (rank by class-mean spatial max, of 1000): kit fox 7, Arctic fox 8, borzoi 56, warplane 104,
speedboat 128, mountain tent 418, obelisk 953, Doberman 761 (and 4th most negative).
Round objects: ranks 111 to 557 on the negative side (only mildly negative).

## What the unit looks like

* Positive pole, class means: Persian cat, papillon, red fox, Pomeranian, tiger cat, tabby, kit fox,
  Arctic fox, Japanese spaniel, beach wagon, lesser panda, lynx, hamster, convertible, weasel, dhole,
  pickup, Pembroke, minivan, Samoyed, ambulance, limousine. Car fronts are a consistent secondary mode.
* Negative pole, class means: bull mastiff, black-and-tan coonhound, pug, Doberman, Rottweiler, boxer,
  Gordon setter, Rhodesian ridgeback, French bulldog, ... golf ball (19th). Strongest negative crops are
  front-facing dog heads with drooping ears or broad round muzzles, plus a few bird heads.
* Feature vis: positive = cat faces; negative = round blobby textures.
* Synthetic opening-angle tuning peaks at 140 to 150 degrees, not at a sharp V.

## Reading

Supports the essay's core claim that 4e:55 is not an arbitrary cat + car mixture: both poles are
coherent and the negative pole names the distinction (upright pointed-ear / tapering heads and car
fronts vs drooping-ear, round-muzzled dog heads). Does not support the literal "V-shape detector"
label: abstract V/O drawings barely drive it, geometric pointed objects (obelisk, tent) do not, and
pointed-snout Doberman sits on the negative pole. A better label is closer to "pointed vs rounded
head silhouette, mostly ears".

Next sharper test (preregister first): erect-eared vs drop-eared dog breeds not yet inspected, and
ear-occlusion edits on the top images.
