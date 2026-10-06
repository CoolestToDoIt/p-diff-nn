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
- [Locked evaluation](06_Locked_Evaluation.md): awaiting authorization for official test and held-out evaluation.

Each substantial phase gets a plan specifying scope, compute limits, deliverables,
validation gates, and authorization status before work begins. Approval of one
phase does not authorize all subsequent training. Do not use official test accuracy
for tuning or candidate selection.
