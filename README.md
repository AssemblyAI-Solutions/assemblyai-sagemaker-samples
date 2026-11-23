# AssemblyAI on Amazon SageMaker - samples and specifications

Reference material for AssemblyAI's speech-to-text model packages on AWS Marketplace. Each product
has its own folder with a sample notebook, runnable examples, sample audio and the exact output each
sample produces.

| Product | What it does | Folder |
|---------|--------------|--------|
| **STT 3.7 Realtime Agent** | Version 3.7.0 of the realtime listing: live speech-to-text for voice agents, with built-in turn detection or turn endings from your own detector, plus whole-file and batch transcription on the same endpoint | [`stt-3-7-realtime-agent/`](stt-3-7-realtime-agent) |
| **Universal-3.6 Pro Streaming** | Live speech-to-text over SageMaker bidirectional streaming, plus whole-file and batch transcription on the same endpoint | [`universal-3-6-pro-streaming/`](universal-3-6-pro-streaming) |
| **Universal 3.5 Pro Sync** | Whole-file speech-to-text over real-time inference and batch transform | [`universal-3-5-pro-sync/`](universal-3-5-pro-sync) |

Start with the folder's README, then the notebook. Both products need a Marketplace subscription,
a SageMaker execution role and a service quota for the instance type you deploy on.

## Use it with a coding agent

[`skills/assemblyai-sagemaker/`](skills/assemblyai-sagemaker) is an agent skill that deploys, calls, scales and
troubleshoots these endpoints. Subscribe on AWS Marketplace and copy the model package ARN first; the agent can't
subscribe for you.

- Claude Code: `/plugin marketplace add AssemblyAI-Solutions/assemblyai-sagemaker-samples`, then
  `/plugin install assemblyai-sagemaker@assemblyai-sagemaker`
- Cursor, Copilot and other agents that read skills: `npx skills add AssemblyAI-Solutions/assemblyai-sagemaker-samples`

Then start a new session and describe what you want, for example "deploy STT 3.7 Realtime Agent in us-west-2 and
stream a test file".

Licensed under the terms in [LICENSE](LICENSE).
