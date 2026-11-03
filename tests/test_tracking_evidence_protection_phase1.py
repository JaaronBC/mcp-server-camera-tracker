from __future__ import annotations

import json
import tempfile
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from agents.tracking.evidence_protection import EvidenceProtectionLock, load_active_protected_frame_ids, write_lease_atomic
from agents.tracking.evidence_store import TrackingEvidenceStore


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestTrackingEvidenceProtectionPhase1(unittest.TestCase):
    def test_cleanup_keeps_valid_unexpired_leases_even_when_over_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "tracking_evidence"
            refs = root / "refs"
            frames = root / "frames"
            frames.mkdir(parents=True, exist_ok=True)

            for index in [1, 2]:
                frame_id = f"frame-protected-{index}"
                frame_ref = f"evidence://frame/{frame_id}"
                (frames / f"{frame_id}.jpg").write_bytes(b"img")
                _write_json(
                    refs / f"{frame_id}.json",
                    {
                        "kind": "frame",
                        "id": frame_id,
                        "reference": frame_ref,
                        "created_at": float(1000 + index),
                        "expires_at": 9000.0,
                        "status": "available",
                        "frame_path": str(frames / f"{frame_id}.jpg"),
                    },
                )
                write_lease_atomic(
                    evidence_root_dir=root,
                    request_id=f"req-protected-{index}",
                    payload={
                        "request_id": f"req-protected-{index}",
                        "frame_reference": frame_ref,
                        "created_at": float(1100 + index),
                        "expires_at": 9000.0,
                    },
                )

            store = TrackingEvidenceStore(
                root,
                max_frame_records=1,
                max_protected_frames=1,
                cleanup_lock_timeout_seconds=0.0,
            )
            try:
                summary = store.cleanup(now_timestamp=2000.0)
            finally:
                store.shutdown()

            self.assertEqual(summary["leases_pruned_overflow"], 0)
            self.assertGreaterEqual(summary["leases_over_budget"], 1)
            self.assertGreaterEqual(summary["capacity_blocked_all_protected"], 1)
            self.assertTrue((root / "protection" / "leases" / "req-protected-1.json").is_file())
            self.assertTrue((root / "protection" / "leases" / "req-protected-2.json").is_file())

    def test_cleanup_evicts_unprotected_before_protected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "tracking_evidence"
            refs = root / "refs"
            frames = root / "frames"
            frames.mkdir(parents=True, exist_ok=True)

            protected_frame_id = "frame-protected"
            old_unprotected_frame_id = "frame-unprotected-old"
            new_unprotected_frame_id = "frame-unprotected-new"

            protected_reference = f"evidence://frame/{protected_frame_id}"
            old_unprotected_reference = f"evidence://frame/{old_unprotected_frame_id}"
            new_unprotected_reference = f"evidence://frame/{new_unprotected_frame_id}"

            for frame_id in [protected_frame_id, old_unprotected_frame_id, new_unprotected_frame_id]:
                (frames / f"{frame_id}.jpg").write_bytes(b"img")

            _write_json(
                refs / f"{protected_frame_id}.json",
                {
                    "kind": "frame",
                    "id": protected_frame_id,
                    "reference": protected_reference,
                    "created_at": 1000.0,
                    "expires_at": 9000.0,
                    "status": "available",
                    "frame_path": str(frames / f"{protected_frame_id}.jpg"),
                },
            )
            _write_json(
                refs / f"{old_unprotected_frame_id}.json",
                {
                    "kind": "frame",
                    "id": old_unprotected_frame_id,
                    "reference": old_unprotected_reference,
                    "created_at": 1001.0,
                    "expires_at": 9000.0,
                    "status": "available",
                    "frame_path": str(frames / f"{old_unprotected_frame_id}.jpg"),
                },
            )
            _write_json(
                refs / f"{new_unprotected_frame_id}.json",
                {
                    "kind": "frame",
                    "id": new_unprotected_frame_id,
                    "reference": new_unprotected_reference,
                    "created_at": 1002.0,
                    "expires_at": 9000.0,
                    "status": "available",
                    "frame_path": str(frames / f"{new_unprotected_frame_id}.jpg"),
                },
            )

            write_lease_atomic(
                evidence_root_dir=root,
                request_id="req-protected",
                payload={
                    "request_id": "req-protected",
                    "frame_reference": protected_reference,
                    "created_at": 1005.0,
                    "expires_at": 9000.0,
                },
            )

            store = TrackingEvidenceStore(
                root,
                max_frame_records=2,
                max_protected_frames=1,
                cleanup_lock_timeout_seconds=0.0,
            )
            try:
                summary = store.cleanup(now_timestamp=1500.0)
            finally:
                store.shutdown()

            self.assertGreaterEqual(summary["removed_capacity"], 1)
            self.assertTrue((refs / f"{protected_frame_id}.json").is_file())
            self.assertTrue((frames / f"{protected_frame_id}.jpg").is_file())
            self.assertFalse((refs / f"{old_unprotected_frame_id}.json").is_file())
            self.assertFalse((frames / f"{old_unprotected_frame_id}.jpg").is_file())
            self.assertTrue((refs / f"{new_unprotected_frame_id}.json").is_file())

    def test_cleanup_skips_when_lock_is_contended(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "tracking_evidence"
            refs = root / "refs"
            frames = root / "frames"
            frames.mkdir(parents=True, exist_ok=True)

            for index in [1, 2]:
                frame_id = f"frame-{index}"
                (frames / f"{frame_id}.jpg").write_bytes(b"img")
                _write_json(
                    refs / f"{frame_id}.json",
                    {
                        "kind": "frame",
                        "id": frame_id,
                        "reference": f"evidence://frame/{frame_id}",
                        "created_at": float(1000 + index),
                        "expires_at": 9000.0,
                        "status": "available",
                        "frame_path": str(frames / f"{frame_id}.jpg"),
                    },
                )

            store = TrackingEvidenceStore(
                root,
                max_frame_records=1,
                max_protected_frames=0,
                cleanup_lock_timeout_seconds=0.0,
            )
            lock = EvidenceProtectionLock(root)
            self.assertTrue(lock.acquire(timeout_seconds=0.0))
            try:
                summary = store.cleanup(now_timestamp=2000.0)
            finally:
                lock.release()

            self.assertEqual(summary["lock_skipped"], 1)
            self.assertEqual(summary["removed_capacity"], 0)
            self.assertEqual(len(list(refs.glob("*.json"))), 2)

            try:
                unlocked = store.cleanup(now_timestamp=2000.0)
            finally:
                store.shutdown()
            self.assertEqual(unlocked["lock_skipped"], 0)
            self.assertGreaterEqual(unlocked["removed_capacity"], 1)

    def test_async_writer_does_not_leave_orphan_jpeg_after_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "tracking_evidence"

            def _slow_real_write(path: str, _frame: object) -> bool:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_bytes(b"jpeg-bytes")
                time.sleep(0.2)
                return True

            fake_cv2 = types.SimpleNamespace(imwrite=_slow_real_write)

            with patch.dict("sys.modules", {"cv2": fake_cv2}):
                store = TrackingEvidenceStore(
                    root,
                    max_age_seconds=1.0,
                    max_frame_records=2,
                    max_protected_frames=0,
                    cleanup_lock_timeout_seconds=0.0,
                )
                frame_record = store.register_frame(
                    frame=object(),
                    timestamp=time.time(),
                    width=8,
                    height=8,
                )

                time.sleep(0.02)
                store.cleanup(now_timestamp=time.time() + 5.0)
                time.sleep(0.25)
                store.shutdown()

            frame_path = root / "frames" / f"{frame_record.frame_id}.jpg"
            metadata_path = root / "refs" / f"{frame_record.frame_id}.json"
            self.assertFalse(metadata_path.is_file())
            self.assertFalse(frame_path.is_file())


    def test_register_frame_rejects_when_capacity_full_and_all_slots_protected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "tracking_evidence"
            refs = root / "refs"
            frames = root / "frames"
            frames.mkdir(parents=True, exist_ok=True)

            existing_frame_id = "frame-protected"
            existing_frame_ref = f"evidence://frame/{existing_frame_id}"
            future_expiry = time.time() + 3600.0
            (frames / f"{existing_frame_id}.jpg").write_bytes(b"img")
            _write_json(
                refs / f"{existing_frame_id}.json",
                {
                    "kind": "frame",
                    "id": existing_frame_id,
                    "reference": existing_frame_ref,
                    "created_at": time.time(),
                    "expires_at": future_expiry,
                    "status": "available",
                    "frame_path": str(frames / f"{existing_frame_id}.jpg"),
                },
            )
            write_lease_atomic(
                evidence_root_dir=root,
                request_id="req-protected",
                payload={
                    "request_id": "req-protected",
                    "frame_reference": existing_frame_ref,
                    "created_at": time.time(),
                    "expires_at": future_expiry,
                },
            )

            store = TrackingEvidenceStore(
                root,
                max_frame_records=1,
                max_protected_frames=1,
                cleanup_lock_timeout_seconds=0.0,
            )
            try:
                frame_record = store.register_frame(
                    frame=object(),
                    timestamp=time.time(),
                    width=8,
                    height=8,
                )
            finally:
                store.shutdown()

            self.assertEqual(frame_record.status, "dropped_storage_full_protected")
            self.assertEqual(len(list(refs.glob("*.json"))), 1)
            self.assertFalse((refs / f"{frame_record.frame_id}.json").is_file())
            self.assertFalse((frames / f"{frame_record.frame_id}.jpg").is_file())

    def test_load_active_protected_frame_ids_prunes_request_id_filename_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "tracking_evidence"
            refs = root / "refs"
            frames = root / "frames"
            leases = root / "protection" / "leases"
            frames.mkdir(parents=True, exist_ok=True)
            leases.mkdir(parents=True, exist_ok=True)

            frame_id = "frame-1"
            frame_ref = f"evidence://frame/{frame_id}"
            (frames / f"{frame_id}.jpg").write_bytes(b"img")
            _write_json(
                refs / f"{frame_id}.json",
                {
                    "kind": "frame",
                    "id": frame_id,
                    "reference": frame_ref,
                    "created_at": 1000.0,
                    "expires_at": 9000.0,
                    "status": "available",
                    "frame_path": str(frames / f"{frame_id}.jpg"),
                },
            )

            mismatch_lease = leases / "req-file-name.json"
            mismatch_lease.write_text(
                json.dumps(
                    {
                        "request_id": "different-request-id",
                        "frame_reference": frame_ref,
                        "created_at": 1001.0,
                        "expires_at": 9000.0,
                    }
                ),
                encoding="utf-8",
            )

            protected, stats = load_active_protected_frame_ids(
                evidence_root_dir=root,
                now_timestamp=2000.0,
                max_protected_frames=10,
            )

            self.assertEqual(protected, set())
            self.assertGreaterEqual(stats["leases_pruned_malformed"], 1)
            self.assertFalse(mismatch_lease.is_file())


    def test_concurrent_register_frame_does_not_exceed_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "tracking_evidence"

            def _quick_write(path: str, _frame: object) -> bool:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_bytes(b"jpeg")
                return True

            fake_cv2 = types.SimpleNamespace(imwrite=_quick_write)

            with patch.dict("sys.modules", {"cv2": fake_cv2}):
                store = TrackingEvidenceStore(
                    root,
                    max_frame_records=1,
                    max_protected_frames=0,
                    cleanup_lock_timeout_seconds=0.05,
                )
                try:
                    def _register_once() -> str:
                        record = store.register_frame(
                            frame=object(),
                            timestamp=time.time(),
                            width=8,
                            height=8,
                        )
                        return record.status

                    with ThreadPoolExecutor(max_workers=8) as executor:
                        statuses = list(executor.map(lambda _: _register_once(), range(8)))

                    self.assertEqual(len(statuses), 8)
                    refs = list((root / "refs").glob("*.json"))
                    self.assertLessEqual(len(refs), 1)
                finally:
                    store.shutdown()

    def test_cleanup_skips_while_writer_finalizes_metadata_under_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "tracking_evidence"
            entered_final_write = threading.Event()

            def _real_write(path: str, _frame: object) -> bool:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_bytes(b"jpeg")
                return True

            fake_cv2 = types.SimpleNamespace(imwrite=_real_write)

            from agents.tracking import evidence_store as evidence_store_module

            original_json_write = evidence_store_module._json_write

            def _slow_final_json_write(path: Path, payload: dict[str, object]) -> None:
                if payload.get("kind") == "frame" and payload.get("status") == "available":
                    entered_final_write.set()
                    time.sleep(0.15)
                original_json_write(path, payload)

            with patch.dict("sys.modules", {"cv2": fake_cv2}), patch.object(
                evidence_store_module,
                "_json_write",
                side_effect=_slow_final_json_write,
            ):
                store = TrackingEvidenceStore(
                    root,
                    max_frame_records=2,
                    max_protected_frames=0,
                    cleanup_lock_timeout_seconds=0.0,
                )
                try:
                    frame_record = store.register_frame(
                        frame=object(),
                        timestamp=time.time(),
                        width=8,
                        height=8,
                    )

                    self.assertTrue(entered_final_write.wait(timeout=1.0))
                    summary = store.cleanup(now_timestamp=time.time())
                    time.sleep(0.2)
                    metadata = json.loads((root / "refs" / f"{frame_record.frame_id}.json").read_text(encoding="utf-8"))
                finally:
                    store.shutdown()

            self.assertEqual(summary["lock_skipped"], 1)
            self.assertEqual(metadata.get("status"), "available")


if __name__ == "__main__":
    unittest.main()
