from scripts import run_p30_search_scaling_anomaly_gate as anomaly
from scripts import run_p30_search_scaling_cross_mode_gates as cross_mode
from scripts import run_p30_search_scaling_promotion_gates as promotion


def test_p30_wrappers_share_campaign_identity() -> None:
    assert promotion.CAMPAIGN == anomaly.CAMPAIGN == cross_mode.CAMPAIGN
    assert promotion.OUTPUT == anomaly.GATE_DIR
    assert anomaly.OUTPUT == cross_mode.ANOMALY_OUTPUT


def test_p30_parent_uses_composed_critic_checkpoint() -> None:
    assert promotion.PARENT.name == "composed"


def test_p30_wrappers_reapply_configuration_after_shared_module_mutation() -> None:
    promotion.gates.PARENT = promotion.ROOT / "wrong"
    anomaly.gate.CAMPAIGN = anomaly.ROOT / "wrong"
    cross_mode.gates.OUTPUT = cross_mode.ROOT / "wrong"
    promotion.configure()
    anomaly.configure()
    cross_mode.configure()
    assert promotion.gates.PARENT == promotion.PARENT
    assert anomaly.gate.CAMPAIGN == anomaly.CAMPAIGN
    assert cross_mode.gates.OUTPUT == cross_mode.OUTPUT
