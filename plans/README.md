# Project plans

Keep project plans, authorization requests, and experiment decisions here.

- [Master architecture and research plan](P_Diff_Project_Plan.md)
- [Foundation work and results](01_Foundation.md): authorized by the request to start; implementation and synthetic verification only.
- [Source checkpoint pilot](02_Source_Checkpoint_Pilot.md): authorized and complete.
- [Pilot results](02_Source_Pilot_Results.md): baseline, checkpoint diversity, and timing.
- [Full source collection](03_Full_Source_Collection.md): authorized and complete.
- [Full collection results](03_Source_Collection_Results.md): 200 checkpoints; integrity and accuracy gates passed.
- [Autoencoder pilot](04_Autoencoder_Pilot.md): authorized and complete; reconstruction gate passed.
- [Autoencoder results](04_Autoencoder_Results.md): 94.14% reconstructed median validation accuracy.
- [Diffusion pilot](05_Diffusion_Pilot.md): authorized and complete.
- [Diffusion results](05_Diffusion_Results.md): target passed; Gaussian and weight averaging performed better.
- [Locked evaluation](06_Locked_Evaluation.md): authorized and complete.
- [Locked results](06_Locked_Evaluation_Results.md): diffusion 77% reliability; target missed; prototype complete.
- [Instability diagnosis](07_Instability_Diagnosis.md): authorized and complete; no new training or test access.
- [Diagnosis results](07_Diagnosis_Results.md): one dominant latent direction and large diffusion extrapolation.
- [Scalar latent pilot](08_Reduced_Latent_Pilot.md): authorized and complete.
- [Scalar results and architecture](08_Scalar_Pilot_Results.md): stable 94.15% median; no advantage over simple sampling.
- [Representation diagnosis](09_Representation_Diagnosis.md): awaiting approval; training/validation only, no training updates.

Each substantial phase gets a plan specifying scope, compute limits, deliverables,
validation gates, and authorization status before work begins. Approval of one
phase does not authorize all subsequent training. Do not use official test accuracy
for tuning or candidate selection.
