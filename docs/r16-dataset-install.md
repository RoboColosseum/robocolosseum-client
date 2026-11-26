# Install the paired R16 data feature

Use the client and policy-server PRs together. This is an opt-in data feature, not a live deployment command. Keep the previous environments for rollback. Do not change running services while installing or testing an isolated environment.

## Dependencies

Use Python 3.11 for the YAM Linux runtime. Install the client checkout with `python -m pip install -e ".[yam,gui]"`. Install the paired policy-server checkout with its existing backend extra (for example `[dreamlabs]`); retain the upstream mutually exclusive agent/CaP-X extra rules. The client needs ffmpeg and ffprobe on PATH. The pinned I2RT SDK already declares `mujoco>=3.3.5`; retain the existing SDK installation procedure rather than adding another client extra or changing unrelated dependency locks.

Install I2RT from a separate clean checkout at `120c3c81400171174604e503943f8d1ebc891058`, apply `patches/i2rt-feedback-history.patch` after `git apply --check`, and install that checkout into the client environment. Do not run motor configuration, calibration, or CAN reset commands as part of installation. See r16-dataset-changes.md for exact capability and control boundaries.

## Setup and preflight

Create a private setup file using `python -m colosseum_client.roboeval_setup /absolute/path/setup.json --template --config /absolute/path/robot.yaml`. This reads configuration and SDK model resources without opening devices. Fill actual arm transforms, camera identities and measured gripper mappings; do not copy test fixture geometry. Mark calibration verified only after it has actually been verified. Validate with `python -m colosseum_client.roboeval_setup /absolute/path/setup.json`.

In a copy of the existing robot YAML, set `recording: true`, `roboeval_setup: /absolute/path/setup.json`, `roboeval_state_hz: 100`, and `skip_upload: true` for acceptance. Keep existing CAN identifiers and model/backend settings. Record capture at 100 Hz; the physical training delivery is 30 Hz. Do not alter control_hz to match the recording frequency. Optional `image_encoding: png` is for the paired local policy server; default remains raw RGB. Continuous RGB recording cannot be combined with depth inference.

Before device construction, the client checks the SDK snapshot/history methods. Missing capabilities abort this opt-in workflow. Source collection errors are recorded as incomplete and must not be presented as valid datasets.

## Existing operator workflow

Use the repository's existing eval GUI or recording CLI and `scripts/review_evals.py`; the independent station UI is not included. On the review page, enter A/B success scores, preference, optional feedback, task difficulty and Drop/Damage/Spill. Events are keyed by immutable run ID. Unknown/not-applicable are explicit choices. Save locally does not contact the Router. Existing Submit/Discard buttons retain their external side effects; they are not part of offline tests or implicitly authorized by installation.

Default exports use `<run>/lerobot`. When an older export exists, the new physical profile is placed at `<run>/export_physical_30hz/lerobot`. Difficulty locks cover both. Export selection prefers a completed physical profile and refuses to fall back to an older package after a failed/incomplete new export. The existing evaluation-ID backfill uses the newly rebuilt path. Existing files are preserved.

## Acceptance before merge

Run standalone client and policy tests, paired wire/image checks, fake-device A/B recording and export, and real r16 dataset readback. Then perform a separately authorized hardware A/B trial using the isolated candidate. Check actual feedback, complete action evidence, timing, videos, package inventory and standard LeRobot readback. Offline tests cannot certify hardware timing, geometry or physical safety. Keep both PRs unmerged until this evidence and repository approval are available.
