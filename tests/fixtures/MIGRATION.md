Independent station Web UI is outside this PR. Its original tests remain in the r16 source project; they are not counted as passing here. Excluded UI-only functions/helpers:

- test_roboeval.py:test_web_backfill_validates_and_reverifies_the_same_extension
- test_roboeval.py:web_upload_functions
- test_roboeval_shared.py:test_actual_web_event_endpoint_only_records_and_requires_active_run
- test_roboeval_shared.py:test_actual_web_submit_keeps_original_optional_feedback_rules
- test_roboeval_shared.py:test_web_operator_events_use_announced_actual_run_and_manifest
- test_roboeval_shared.py:web_session

Repository review UI is tested by test_review_b6.py. Dataset backfill selection is tested by test_dataset_paths.py.

The optional-recording tests exclude the independent `web/app.py` route test. Stop tests exercise the repository's `scripts/eval_gui.py`, retaining upstream SO101 behavior and testing YAM's skipped-execution marker. Fixtures include the upstream task-difficulty answer and the export function's returned path.

- test_roboeval_compact.py:test_actual_web_validator_and_reverification_accept_compact_layout (standalone Web service only; client real upload protocol test retained)

Site-only yam2 hardware aliases and the independent UI SQLite completion-date uploader are excluded; the original yam identity and evaluation-ID uploader remain.
