"""Unit tests for jupyter_mcp package."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import nbformat

from jupyter_mcp import (
    ConflictError,
    _utc_now,
    _new_id,
    _tool_error,
    _strip_ansi,
    _truncate_text,
    _format_display_dict,
    _parse_iopub_messages,
    _build_nbformat_outputs,
    _OUTPUT_TEXT_LIMIT,
)
from jupyter_mcp.kernel import SessionRecord
from jupyter_mcp.notebooks import FileNotebookStore, parse_cell_selector
from jupyter_mcp.operations import OperationRecord, OperationManager
from jupyter_mcp.orchestrator import ExecutionOrchestrator


# ---------------------------------------------------------------------------
# Helpers / __init__
# ---------------------------------------------------------------------------


class TestHelpers(unittest.TestCase):

    def test_utc_now_returns_float(self):
        t = _utc_now()
        self.assertIsInstance(t, float)
        self.assertAlmostEqual(t, time.time(), delta=1.0)

    def test_new_id_format(self):
        ident = _new_id("sess")
        self.assertTrue(ident.startswith("sess_"), ident)
        suffix = ident[len("sess_"):]
        self.assertEqual(len(suffix), 10)
        self.assertTrue(all(c in "0123456789abcdef" for c in suffix), suffix)

    def test_new_id_uniqueness(self):
        ids = {_new_id("x") for _ in range(100)}
        self.assertEqual(len(ids), 100)

    def test_tool_error_no_details(self):
        err = _tool_error("NotFound", "thing missing")
        self.assertEqual(err, {"error": {"code": "NotFound", "message": "thing missing"}})

    def test_tool_error_with_details(self):
        err = _tool_error("Conflict", "rev mismatch", {"current": "abc"})
        self.assertEqual(err["error"]["details"], {"current": "abc"})

    def test_strip_ansi(self):
        text = "\x1b[31mred\x1b[0m plain"
        self.assertEqual(_strip_ansi(text), "red plain")

    def test_strip_ansi_no_op(self):
        text = "no escapes here"
        self.assertEqual(_strip_ansi(text), text)

    def test_truncate_text_under_limit(self):
        text = "short"
        result, was_truncated = _truncate_text(text, 100)
        self.assertEqual(result, text)
        self.assertFalse(was_truncated)

    def test_truncate_text_at_limit(self):
        text = "a" * 100
        result, was_truncated = _truncate_text(text, 100)
        self.assertEqual(result, text)
        self.assertFalse(was_truncated)

    def test_truncate_text_over_limit(self):
        text = "a" * 200
        result, was_truncated = _truncate_text(text, 100)
        self.assertTrue(was_truncated)
        self.assertIn("truncated", result)
        self.assertTrue(result.startswith("a" * 100))

    def test_format_display_dict_text_only(self):
        d = _format_display_dict({"text/plain": "hello"})
        self.assertEqual(d, {"text": "hello"})

    def test_format_display_dict_images_excluded(self):
        d = _format_display_dict({"image/png": "base64data=="}, include_images=False)
        self.assertIn("image_png", d)
        self.assertIn("[base64 PNG", d["image_png"])

    def test_format_display_dict_images_included(self):
        d = _format_display_dict({"image/png": "base64data=="}, include_images=True)
        self.assertEqual(d["image_png"], "base64data==")

    def test_format_display_dict_html_truncation(self):
        long_html = "x" * 3000
        d = _format_display_dict({"text/html": long_html})
        self.assertIn("truncated", d["html"])
        self.assertLessEqual(len(d["html"]), 2020)

    def test_format_display_dict_unknown_mime(self):
        d = _format_display_dict({"application/unknown": "data", "text/plain": "x"})
        self.assertIn("application/unknown", d.get("other_mime_types", []))

    def test_parse_iopub_messages_stdout(self):
        msgs = [{"msg_type": "stream", "content": {"name": "stdout", "text": "hello\n"}}]
        result = _parse_iopub_messages(msgs)
        self.assertEqual(result["stdout"], "hello\n")
        self.assertEqual(result["stderr"], "")
        self.assertIsNone(result["error"])
        self.assertFalse(result["truncated"])

    def test_parse_iopub_messages_stderr(self):
        msgs = [{"msg_type": "stream", "content": {"name": "stderr", "text": "warn"}}]
        result = _parse_iopub_messages(msgs)
        self.assertEqual(result["stderr"], "warn")

    def test_parse_iopub_messages_error(self):
        msgs = [{"msg_type": "error", "content": {
            "ename": "ValueError",
            "evalue": "bad",
            "traceback": ["\x1b[31mTraceback\x1b[0m"],
        }}]
        result = _parse_iopub_messages(msgs)
        self.assertIsNotNone(result["error"])
        self.assertEqual(result["error"]["ename"], "ValueError")
        self.assertNotIn("\x1b", result["error"]["traceback"])

    def test_parse_iopub_messages_display_data(self):
        msgs = [{"msg_type": "display_data", "content": {"data": {"text/plain": "42"}}}]
        result = _parse_iopub_messages(msgs)
        self.assertEqual(len(result["rich_outputs"]), 1)
        self.assertEqual(result["rich_outputs"][0]["type"], "display_data")

    def test_parse_iopub_messages_truncation(self):
        big_text = "x" * (_OUTPUT_TEXT_LIMIT + 1)
        msgs = [{"msg_type": "stream", "content": {"name": "stdout", "text": big_text}}]
        result = _parse_iopub_messages(msgs)
        self.assertTrue(result["truncated"])

    def test_build_nbformat_outputs_stream(self):
        msgs = [{"msg_type": "stream", "content": {"name": "stdout", "text": "hi"}}]
        outputs = _build_nbformat_outputs(msgs, execution_count=1)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["output_type"], "stream")
        self.assertEqual(outputs[0]["text"], "hi")

    def test_build_nbformat_outputs_stream_coalesced(self):
        msgs = [
            {"msg_type": "stream", "content": {"name": "stdout", "text": "a"}},
            {"msg_type": "stream", "content": {"name": "stdout", "text": "b"}},
        ]
        outputs = _build_nbformat_outputs(msgs, execution_count=None)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["text"], "ab")

    def test_build_nbformat_outputs_execute_result(self):
        msgs = [{"msg_type": "execute_result", "content": {
            "data": {"text/plain": "42"}, "metadata": {}, "execution_count": 1
        }}]
        outputs = _build_nbformat_outputs(msgs, execution_count=1)
        self.assertEqual(outputs[0]["output_type"], "execute_result")

    def test_build_nbformat_outputs_clear_output(self):
        msgs = [
            {"msg_type": "stream", "content": {"name": "stdout", "text": "old"}},
            {"msg_type": "clear_output", "content": {"wait": False}},
            {"msg_type": "stream", "content": {"name": "stdout", "text": "new"}},
        ]
        outputs = _build_nbformat_outputs(msgs, execution_count=None)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["text"], "new")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses(unittest.TestCase):

    def test_session_record_defaults(self):
        rec = SessionRecord(
            session_id="sess_abc",
            python_path="python",
            isolation="ephemeral",
            cwd="/tmp",
            created_at=1.0,
            last_used_at=1.0,
        )
        self.assertEqual(rec.state, "ready")

    def test_operation_record_defaults(self):
        rec = OperationRecord(
            op_id="op_abc",
            kind="run_code",
            status="queued",
            submitted_at=1.0,
        )
        self.assertIsNone(rec.started_at)
        self.assertIsNone(rec.ended_at)
        self.assertFalse(rec.cancelled)


# ---------------------------------------------------------------------------
# OperationManager
# ---------------------------------------------------------------------------


class TestOperationManager(unittest.TestCase):

    def setUp(self):
        self.mgr = OperationManager(max_workers=2, max_inflight=4)

    def tearDown(self):
        self.mgr.shutdown()

    def test_submit_and_get_completed(self):
        def job(_op_id):
            return {"answer": 42}

        snap = self.mgr.submit("test", job)
        self.assertIn(snap["status"], {"queued", "running", "completed"})
        result = self.mgr.get(snap["op_id"], wait_ms=5000)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"]["answer"], 42)

    def test_submit_and_get_failed(self):
        def job(_op_id):
            raise RuntimeError("boom")

        snap = self.mgr.submit("test", job)
        result = self.mgr.get(snap["op_id"], wait_ms=5000)
        self.assertEqual(result["status"], "failed")
        self.assertIn("boom", result["error"]["message"])

    def test_conflict_error_maps_to_conflict_code(self):
        def job(_op_id):
            raise ConflictError("Revision conflict: expected abc, current def")

        snap = self.mgr.submit("test", job)
        result = self.mgr.get(snap["op_id"], wait_ms=5000)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "Conflict")
        self.assertIn("Revision conflict", result["error"]["message"])

    def test_plain_runtime_error_maps_to_execution_error(self):
        def job(_op_id):
            raise RuntimeError("something else broke")

        snap = self.mgr.submit("test", job)
        result = self.mgr.get(snap["op_id"], wait_ms=5000)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "ExecutionError")

    def test_get_nonexistent_op(self):
        result = self.mgr.get("op_doesnotexist")
        self.assertIn("error", result)
        self.assertEqual(result["error"]["code"], "NotFound")

    def test_max_inflight_limit(self):
        barrier = threading.Barrier(2)

        def blocking_job(_op_id):
            barrier.wait(timeout=5)
            return {}

        snaps = []
        for _ in range(4):
            snaps.append(self.mgr.submit("test", blocking_job))

        # Fifth should be rejected
        fifth = self.mgr.submit("test", blocking_job)
        self.assertIn("error", fifth)
        self.assertEqual(fifth["error"]["code"], "Busy")

        # Unblock workers
        barrier.abort()

    def test_update_progress(self):
        event = threading.Event()

        def job(op_id):
            self.mgr.update_progress(op_id, {"pct": 50})
            event.set()
            time.sleep(0.1)
            return {}

        snap = self.mgr.submit("test", job)
        event.wait(timeout=5)
        snap2 = self.mgr.get(snap["op_id"])
        # progress may be set while still running
        self.mgr.get(snap["op_id"], wait_ms=2000)  # let it finish

    def test_snapshot_includes_timings(self):
        def job(_op_id):
            return {}

        snap = self.mgr.submit("test", job)
        result = self.mgr.get(snap["op_id"], wait_ms=5000)
        self.assertIn("timings", result)
        self.assertGreaterEqual(result["timings"]["duration_ms"], 0)


# ---------------------------------------------------------------------------
# FileNotebookStore
# ---------------------------------------------------------------------------


class TestFileNotebookStore(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = FileNotebookStore()
        self.nb_path = str(Path(self.tmpdir.name) / "test.ipynb")

    def tearDown(self):
        self.tmpdir.cleanup()

    def _create(self, path=None):
        return self.store.create_notebook(path or self.nb_path, "python3")

    def test_create_and_read_notebook(self):
        result = self._create()
        self.assertEqual(result["status"], "created")
        self.assertIn("revision", result)

        read = self.store.read(self.nb_path, include_outputs=False, output_limit=1000)
        self.assertEqual(read["cell_count"], 0)
        self.assertIn("revision", read)

    def test_create_duplicate_raises_conflict(self):
        self._create()
        with self.assertRaises(FileExistsError):
            self._create()

    def test_insert_cell(self):
        result = self._create()
        rev = result["revision"]
        ins = self.store.insert_cell(self.nb_path, rev, 0, "code", "x = 1")
        self.assertEqual(ins["status"], "inserted")
        self.assertEqual(ins["cell_count"], 1)
        self.assertNotEqual(ins["revision"], rev)

    def test_update_cell(self):
        result = self._create()
        rev = result["revision"]
        ins = self.store.insert_cell(self.nb_path, rev, 0, "code", "x = 1")
        upd = self.store.update_cell(self.nb_path, ins["revision"], 0, "x = 2", True)
        self.assertEqual(upd["status"], "updated")

        read = self.store.read(self.nb_path, include_outputs=False, output_limit=1000)
        self.assertEqual(read["cells"][0]["source"], "x = 2")

    def test_delete_cell(self):
        result = self._create()
        rev = result["revision"]
        ins = self.store.insert_cell(self.nb_path, rev, 0, "code", "x = 1")
        dl = self.store.delete_cell(self.nb_path, ins["revision"], 0)
        self.assertEqual(dl["status"], "deleted")
        self.assertEqual(dl["cell_count"], 0)

    def test_move_cell(self):
        result = self._create()
        rev = result["revision"]
        ins1 = self.store.insert_cell(self.nb_path, rev, 0, "code", "a = 1")
        ins2 = self.store.insert_cell(self.nb_path, ins1["revision"], 1, "code", "b = 2")
        mv = self.store.move_cell(self.nb_path, ins2["revision"], 0, 1)
        self.assertEqual(mv["status"], "moved")

        read = self.store.read(self.nb_path, include_outputs=False, output_limit=1000)
        self.assertEqual(read["cells"][0]["source"], "b = 2")
        self.assertEqual(read["cells"][1]["source"], "a = 1")

    def test_revision_conflict(self):
        result = self._create()
        rev = result["revision"]
        self.store.insert_cell(self.nb_path, rev, 0, "code", "x = 1")
        # Use stale revision — must raise ConflictError (subclass of RuntimeError)
        with self.assertRaises(ConflictError):
            self.store.insert_cell(self.nb_path, rev, 0, "code", "y = 2")

    def test_clear_outputs_single_cell(self):
        result = self._create()
        rev = result["revision"]
        ins = self.store.insert_cell(self.nb_path, rev, 0, "code", "x = 1")
        cl = self.store.clear_outputs(self.nb_path, ins["revision"], 0)
        self.assertEqual(cl["status"], "outputs_cleared")
        self.assertEqual(cl["cleared_cells"], 1)

    def test_clear_outputs_all_cells(self):
        result = self._create()
        rev = result["revision"]
        ins1 = self.store.insert_cell(self.nb_path, rev, 0, "code", "x = 1")
        ins2 = self.store.insert_cell(self.nb_path, ins1["revision"], 1, "code", "y = 2")
        cl = self.store.clear_outputs(self.nb_path, ins2["revision"])
        # cleared_cells counts only cells that had outputs/execution_count;
        # fresh cells start empty, so count is 0, but the operation still succeeds.
        self.assertEqual(cl["status"], "outputs_cleared")
        self.assertEqual(cl["cell_count"], 2)
        self.assertIn("revision", cl)

    def test_batch_cells(self):
        result = self._create()
        rev = result["revision"]
        ops = [
            {"action": "insert", "cell_index": 0, "cell_type": "code", "source": "a = 1"},
            {"action": "insert", "cell_index": 1, "cell_type": "markdown", "source": "# Title"},
        ]
        batch = self.store.batch_cells(self.nb_path, rev, ops)
        self.assertEqual(batch["status"], "batch_applied")
        self.assertEqual(batch["operations_applied"], 2)
        self.assertEqual(batch["cell_count"], 2)

    def test_list_notebooks(self):
        self._create()
        listing = self.store.list_notebooks(self.tmpdir.name)
        self.assertEqual(len(listing["notebooks"]), 1)
        self.assertEqual(listing["notebooks"][0]["name"], "test.ipynb")

    def test_delete_notebook(self):
        result = self._create()
        rev = result["revision"]
        dl = self.store.delete_notebook(self.nb_path, rev)
        self.assertEqual(dl["status"], "deleted")
        self.assertFalse(Path(self.nb_path).exists())

    def test_delete_notebook_revision_conflict(self):
        result = self._create()
        rev = result["revision"]
        ins = self.store.insert_cell(self.nb_path, rev, 0, "code", "x")
        with self.assertRaises(RuntimeError):
            self.store.delete_notebook(self.nb_path, rev)  # stale rev


# ---------------------------------------------------------------------------
# ExecutionOrchestrator
# ---------------------------------------------------------------------------


class TestExecutionOrchestrator(unittest.TestCase):

    def _make_orchestrator(self, execute_return):
        provider = MagicMock()
        provider.execute.return_value = execute_return
        nb_store = MagicMock()
        return ExecutionOrchestrator(provider, nb_store), provider, nb_store

    def test_run_code_summary_capture(self):
        execute_return = {
            "stdout": "hello",
            "stderr": "",
            "rich_outputs": [{"type": "display_data"} for _ in range(15)],
            "error": None,
            "truncated": False,
            "execution_count": 1,
            "_raw_messages": [],
        }
        orch, provider, _ = self._make_orchestrator(execute_return)
        result = orch.run_code("sess_1", "print('hello')", 60, "summary")
        self.assertEqual(result["stdout"], "hello")
        # summary mode caps rich_outputs at 10
        self.assertLessEqual(len(result["rich_outputs"]), 10)

    def test_run_code_full_capture(self):
        execute_return = {
            "stdout": "hello",
            "stderr": "",
            "rich_outputs": [],
            "error": None,
            "truncated": False,
            "execution_count": 1,
            "_raw_messages": [],
        }
        orch, provider, _ = self._make_orchestrator(execute_return)
        result = orch.run_code("sess_1", "print('hello')", 60, "full")
        self.assertEqual(result["stdout"], "hello")

    def test_run_notebook_fresh_uses_notebook_parent_as_cwd(self):
        provider = MagicMock()
        provider.create_session.return_value = SessionRecord(
            session_id="sess_fresh",
            python_path="python",
            isolation="ephemeral",
            cwd="/tmp",
            created_at=1.0,
            last_used_at=1.0,
        )
        provider.execute.return_value = {
            "stdout": "",
            "stderr": "",
            "rich_outputs": [],
            "error": None,
            "truncated": False,
            "execution_count": 1,
            "_raw_messages": [],
        }
        nb_store = MagicMock()
        nb_store.read.return_value = {
            "path": "/tmp/project/notebooks/demo.ipynb",
            "revision": "rev1",
            "cells": [{"cell_type": "code", "source": "print('x')"}],
        }
        nb_store.write_execution_cell.return_value = {"revision": "rev2", "cell_index": 0}
        orch = ExecutionOrchestrator(provider, nb_store)

        result = orch.run_notebook(path="/tmp/project/notebooks/demo.ipynb", mode="fresh")

        self.assertEqual(result["status"], "completed")
        provider.create_session.assert_called_once_with(
            python_path="python",
            cwd="/tmp/project/notebooks",
            isolation="ephemeral",
        )
        provider.close_session.assert_called_once_with("sess_fresh", force=True)
        self.assertEqual(nb_store.read.call_count, 1)

    def test_run_notebook_session_does_not_create_new_session(self):
        provider = MagicMock()
        provider.list_sessions.return_value = [
            SessionRecord(
                session_id="sess_existing",
                python_path="python",
                isolation="persistent",
                cwd="/tmp/project",
                created_at=1.0,
                last_used_at=1.0,
            )
        ]
        provider.execute.return_value = {
            "stdout": "",
            "stderr": "",
            "rich_outputs": [],
            "error": None,
            "truncated": False,
            "execution_count": 1,
            "_raw_messages": [],
        }
        nb_store = MagicMock()
        nb_store.read.return_value = {
            "path": "/tmp/project/notebooks/demo.ipynb",
            "revision": "rev1",
            "cells": [{"cell_type": "code", "source": "print('x')"}],
        }
        nb_store.write_execution_cell.return_value = {"revision": "rev2", "cell_index": 0}
        orch = ExecutionOrchestrator(provider, nb_store)

        result = orch.run_notebook(
            path="/tmp/project/notebooks/demo.ipynb",
            mode="session",
            target_session_id="sess_existing",
        )

        self.assertEqual(result["status"], "completed")
        provider.create_session.assert_not_called()
        provider.close_session.assert_not_called()
        provider.list_sessions.assert_called_once()

    def test_parse_cell_selector_all(self):
        self.assertEqual(parse_cell_selector(None, 5), (0, 5))

    def test_parse_cell_selector_all_keyword(self):
        self.assertEqual(parse_cell_selector("all", 3), (0, 3))

    def test_parse_cell_selector_single(self):
        self.assertEqual(parse_cell_selector("2", 5), (2, 3))

    def test_parse_cell_selector_single_out_of_range(self):
        with self.assertRaises(ValueError):
            parse_cell_selector("99", 5)

    def test_parse_cell_selector_range(self):
        self.assertEqual(parse_cell_selector("2:5", 10), (2, 5))

    def test_parse_cell_selector_open_start(self):
        self.assertEqual(parse_cell_selector(":5", 10), (0, 5))

    def test_parse_cell_selector_open_end(self):
        self.assertEqual(parse_cell_selector("3:", 10), (3, 10))

    def test_parse_cell_selector_clamps(self):
        self.assertEqual(parse_cell_selector("0:99", 5), (0, 5))


if __name__ == "__main__":
    unittest.main()
