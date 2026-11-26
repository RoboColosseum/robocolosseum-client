# R16 dataset migration — implementation in progress

This change is paired with the other RoboColosseum repository. Do not deploy or merge until paired integration tests and an explicitly authorized hardware A/B acceptance pass are recorded.

## Data contract

The main action remains a policy target, not necessarily the final SDK command. Actual sent commands and full model blocks (including unused tails) are distinct evidence. The physical delivery profile uses 30 Hz and three cameras, nine files per run. Validate original evidence before producing the reduced package; the delivery cannot reconstruct removed high-frequency states or frames. Gripper effort is signed motor torque in N.m, not fingertip force. Missing legacy effort is not fabricated.

## Low-level changes requiring review

| Layer | Change | Verification / remaining gate |
| --- | --- | --- |
| I2RT motor feedback | Coherent feedback snapshots, sequence/time metadata, bounded history and explicit drain/start/stop | Pinned patch check/apply and six isolated history/preflight tests passed; no runtime auto-patching |
| YAM adapter | Passive feedback mapping, motor-7 effort, command evidence and source camera metadata | Compare direction, offsets, gripper mapping and command outputs against upstream |
| Control/stop | A distinct skipped-execution marker prevents recording a command the GUI did not send | Preserve valid legacy None returns; test stop before/during/after inference |
| Cameras | Continuous RGB capture and timestamps; retain upstream optional depth | Combined mode is rejected before devices open; separate original depth mode retained |
| Policy protocol | ActionPlan fields 8–11 add raw actions, provenance and action evidence | Paired descriptors, legacy wire compatibility and fake Router mapping tests pass |
| Dream backend | Retain raw replies and interpolation mapping; drain already-requested terminal reply | Compare action arrays with immutable upstream fixture; do not issue extra model requests |

## Upstream preservation

Retain SO101 depth and torque-off confirmation, remote JPEG/ping settings, country metadata, task difficulty, model negotiation and SDK reconnect behavior. These are not replaced with older candidate copies.

## Exclusions and rollback

No independent station UI, private configuration, credentials, real datasets or live deployment scripts are included. Keep previous paired packages and configuration for rollback. Source changes alone do not authorize a service restart, CAN operation, calibration or robot motion.

## Acceptance status

Migration is incomplete. No new candidate hardware run has occurred. Existing r16 field results are reference evidence only. Record final commands/results and paired PR links here before requesting merge.

## SDK patch installation (offline preparation)

Use a separate SDK checkout at `120c3c81400171174604e503943f8d1ebc891058`. Verify `git rev-parse HEAD` and a clean working tree, then run `git apply --check /path/to/client/patches/i2rt-feedback-history.patch` and `git apply /path/to/client/patches/i2rt-feedback-history.patch` from that SDK checkout. Install this checkout into the robot environment. Never patch the running SDK in place. The patch does not change CAN IDs, watchdog configuration or command rate; source review and hardware acceptance remain required. Roll back by selecting the prior environment, not reversing unrelated local modifications.

## Supported camera combinations

Continuous RoboEval RGB capture and depth inference cannot be enabled together. The adapter rejects the combination before constructing devices because the RGB recorder does not retain matched depth frames. Existing depth-only inference and continuous RGB recording remain separate paths; simultaneous support is not claimed.

## Explicitly excluded control change

The r16 five-encoder-count initial-position tolerance is NOT migrated. By user decision, `_execute_clip_commanded` is identical to upstream main: retain its initial-position validation and existing clipping behavior. Only recording instrumentation is added around successful command sends.

## Existing architecture and configuration

The existing GUI reuses `record_evals.py` and `review_evals.py`. B6 events are added to its per-run cards and existing save payload, keyed by run ID. No production API route or Router result field is added. Both review surfaces passed synthetic save/reload checks.

Site-specific `yam2` aliases, the independent SQLite completion-date uploader and extra Router TCP diagnostics are excluded. The original `yam` adapter identity, evaluation-ID upload command and connection lifecycle remain.

New recording settings are `roboeval_setup` (private geometry/calibration) and `roboeval_state_hz` (0 legacy, 100 continuous). Optional local PNG uses the existing `image_encoding` field. Default raw and remote JPEG remain. Dependency extras and lockfile are unchanged; pinned I2RT already supplies MuJoCo.

`export_expanded_run` remains the common export entrypoint; consumers use its returned path. Failed latest exports cannot silently select older data. The unused historical dual-rate export tool and independent station UI are excluded.

## Offline validation limits

Original control tests, recording fault injection, stop-before/during/after-inference, export paths, GUI saving and paired wire/image tests pass. Both wheels build. Existing r16 A/B packages pass the delivery validator and official LeRobot/PyAV readback (1,105/1,355 rows). This validates format/readback, not a hardware run of the migrated code.

Windows regressions have existing Rover preview/POSIX-permission limitations and a policy-server POSIX process-launch/SIGINT limitation. Linux regression and real A/B acceptance remain required before merge.

The Windows full client run reported 645 passed, 4 skipped and the two baseline failures above. After removing site-only aliases/diagnostics/date-upload code, affected suites passed (123 passed/1 skipped, 34 passed, and 36 passed); the final export-retry suite passed 24 tests. Three passive feedback tests also passed. Counts overlap and should not be added.
