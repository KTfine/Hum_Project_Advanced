# Formal evaluation status

Date: 2026-11-26, Asia/Shanghai.

The obsolete HumanML3D/BoLD pilot evidence and generated cache artifacts were deleted at the user's request. No metrics from that pilot remain part of the deliverable.

The replacement evaluation is complete:

- Penn Action: 60 natural-video samples for activity and 13-joint pose.
- RAVDESS: 56 balanced video-only samples for seven-class emotion recognition.
- No training, human judging, LLM judging, or post-result sample removal.
- Penn Action completed real inference for DeepSeek, X-CLIP large/base, and
  YOLO11-pose x/l/m/s/n. The frozen run is
  `D:\AI_cache\hum_project_evaluation\runs\20261126-penn-formal`.
- RAVDESS completed real inference for DeepSeek and HSEmotion B2/B0. The frozen
  run is `D:\AI_cache\hum_project_evaluation\runs\20261126-ravdess-formal`.
- Every formal model has a full manifest denominator and zero failed samples.
- The combined results and bounded claim are recorded in
  `experiments/records/2026-11-26-human-understanding-formal.md`.
