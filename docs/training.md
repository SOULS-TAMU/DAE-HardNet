# Training, saving and deployment

DAE training interfaces live in daehn.dae; reusable backbones and projected training utilities live in daehn.train. Separate supervised, derivative and physics losses where your formulation requires them. Training through projection and projecting a pretrained prediction are different experimental protocols.

Top-level daehn.save_predictor and daehn.load_predictor dispatch to the serialization module. The NPZ PyTree helpers in daehn.serialization.weights save arrays with explicit structure metadata. A reload should be tested on the same inputs and dtype before deploying a predictor.

Legacy benchmark parameter_tree.pkl files remain benchmark artifacts. Load pickle only from trusted sources. A checkpoint hash verifies byte identity, not trust. Saving weights alone does not record all physical parameters, architecture, normalization and solver settings needed to reconstruct predictions.

Export/runtime APIs provide portable and native executable paths; their compatibility depends on JAX versions, export support and the deployment platform. The initial CPU CI does not certify every export target. See daehn.export and daehn.runtime and validate target-specific outputs before deployment.

Complete case-study training and control scripts remain in daehn-benchmarks. This library contains documentation snippets and regression tests rather than large datasets or saved manuscript models.
