from threading import Thread
from time import sleep

from scripts.run_search_scaling_iteration2 import wait_for_file


def test_wait_for_file_accepts_a_delayed_dependency(tmp_path) -> None:
    dependency = tmp_path / "ready.status"

    def create_dependency() -> None:
        sleep(0.02)
        dependency.write_text("complete\n", encoding="utf-8")

    thread = Thread(target=create_dependency)
    thread.start()
    wait_for_file(dependency, poll_seconds=0.01, stage="test")
    thread.join()
    assert dependency.read_text(encoding="utf-8").strip() == "complete"
