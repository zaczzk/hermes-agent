from concurrent.futures import ThreadPoolExecutor

from hermes_state import SessionDB
from hermes_state_store_identity import get_store_id


def test_store_id_is_stable_across_handles_and_restart(tmp_path):
    path = tmp_path / "state.db"
    first = SessionDB(path)
    try:
        store_id = get_store_id(first)
        assert store_id.startswith("hermes-state-v1:")
        assert get_store_id(first) == store_id
    finally:
        first.close()

    reopened = SessionDB(path)
    try:
        assert get_store_id(reopened) == store_id
    finally:
        reopened.close()


def test_store_id_changes_when_database_is_recreated(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    original = get_store_id(db)
    db.close()
    for candidate in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
        candidate.unlink(missing_ok=True)

    replacement = SessionDB(path)
    try:
        assert get_store_id(replacement) != original
    finally:
        replacement.close()


def test_concurrent_openers_converge_on_one_store_id(tmp_path):
    path = tmp_path / "state.db"

    def open_and_read(_index):
        db = SessionDB(path)
        try:
            return get_store_id(db)
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        identities = set(pool.map(open_and_read, range(8)))
    assert len(identities) == 1
