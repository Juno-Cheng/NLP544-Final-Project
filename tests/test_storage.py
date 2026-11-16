import json
import threading

import pytest

from trustsim.storage import RecordLog, read_jsonl, write_jsonl


def key(record: dict) -> str:
    return record["id"]


def test_jsonl_round_trip_keeps_non_ascii_text(tmp_path):
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [{"id": "a", "text": "Grüße"}])
    assert "Grüße" in path.read_text(encoding="utf-8")
    assert read_jsonl(path) == [{"id": "a", "text": "Grüße"}]


def test_latest_record_for_a_key_wins(tmp_path):
    log = RecordLog(tmp_path / "log.jsonl", key)
    log.append({"id": "a", "status": "error"})
    log.append({"id": "b", "status": "ok"})
    log.append({"id": "a", "status": "ok"})
    assert log.latest() == {"a": {"id": "a", "status": "ok"}, "b": {"id": "b", "status": "ok"}}


def test_missing_log_reads_as_empty(tmp_path):
    assert RecordLog(tmp_path / "none.jsonl", key).latest() == {}


def test_partial_last_line_from_a_killed_run_is_ignored(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text(json.dumps({"id": "a"}) + "\n" + '{"id": "b", "sta', encoding="utf-8")
    assert set(RecordLog(path, key).latest()) == {"a"}


def test_corrupt_line_before_the_end_is_an_error(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"id": \n' + json.dumps({"id": "a"}) + "\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        RecordLog(path, key).latest()


def test_appends_from_many_threads_are_all_kept(tmp_path):
    log = RecordLog(tmp_path / "log.jsonl", key)
    threads = [
        threading.Thread(target=lambda n=n: [log.append({"id": f"{n}-{i}"}) for i in range(50)])
        for n in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(log.latest()) == 400
