# AssemblyAI on Amazon SageMaker - samples and specifications

Reference material for AssemblyAI's speech-to-text model packages on AWS Marketplace. Each product
has its own folder with a sample notebook, runnable examples, sample audio and the exact output each
sample produces.

| Product | What it does | Folder |
|---------|--------------|--------|
| **Universal-3.6 Pro Streaming** | Live speech-to-text over SageMaker bidirectional streaming, plus whole-file and batch transcription on the same endpoint | [`universal-3-6-pro-streaming/`](universal-3-6-pro-streaming) |
| **Universal 3.5 Pro Sync** | Whole-file speech-to-text over real-time inference and batch transform | [`universal-3-5-pro-sync/`](universal-3-5-pro-sync) |

Start with the folder's README, then the notebook. Both products need a Marketplace subscription,
a SageMaker execution role and a service quota for the instance type you deploy on.

Licensed under the terms in [LICENSE](LICENSE).
