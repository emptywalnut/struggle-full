from threading import Thread
from time import sleep

import scripts.run_search_scaling_iteration2_promotion_gates as wrapper


def test_promotion_wrapper_waits_for_successful_pressure_status(tmp_path, monkeypatch) -> None:
    status = tmp_path / "pressure.status"
    monkeypatch.setattr(wrapper, "PRESSURE_D8_STATUS", status)

    def mark_complete() -> None:
        sleep(0.02)
        status.write_text("complete\n", encoding="utf-8")

    thread = Thread(target=mark_complete)
    thread.start()
    wrapper.wait_for_pressure_d8(poll_seconds=0.01)
    thread.join()


def test_promotion_wrapper_rejects_failed_pressure_status(tmp_path, monkeypatch) -> None:
    status = tmp_path / "pressure.status"
    status.write_text("failed rc=137\n", encoding="utf-8")
    monkeypatch.setattr(wrapper, "PRESSURE_D8_STATUS", status)
    try:
        wrapper.wait_for_pressure_d8(poll_seconds=0.01)
    except RuntimeError as error:
        assert "failed rc=137" in str(error)
    else:
        raise AssertionError("failed pressure adjudication was accepted")
