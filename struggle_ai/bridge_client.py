from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRIDGE = ROOT / "bridge" / "saito_bridge.js"


class BridgeError(RuntimeError):
    pass


class TwilightBridgeClient:
    """JSON-lines client for the Node headless bridge."""

    def __init__(self, bridge_path: Path | str | None = DEFAULT_BRIDGE):
        self.bridge_path = Path(bridge_path or DEFAULT_BRIDGE)
        self._next_id = 1
        self.proc = subprocess.Popen(
            ["node", str(self.bridge_path)],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self) -> "TwilightBridgeClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def call(self, cmd: str, **payload: Any) -> Any:
        if self.proc.poll() is not None:
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
            raise BridgeError(f"bridge exited with code {self.proc.returncode}: {stderr}")
        msg = {"id": self._next_id, "cmd": cmd, **payload}
        self._next_id += 1
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
            raise BridgeError(f"bridge produced no response: {stderr}")
        response = json.loads(line)
        if not response.get("ok"):
            raise BridgeError(response.get("error", "unknown bridge error"))
        return response["result"]

    def reset(self, seed: int = 1) -> dict[str, Any]:
        return self.call("reset", seed=seed)

    def legal_actions(self) -> list[dict[str, Any]]:
        return self.call("legal_actions")

    def step(self, action: dict[str, Any], forced_rolls: list[int | float] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": action}
        if forced_rolls is not None:
            payload["forced_rolls"] = forced_rolls
        return self.call("step", **payload)

    def snapshot(
        self,
        *,
        runtime_checkpoint: bool = True,
        include_actions: bool = True,
    ) -> dict[str, Any]:
        return self.call(
            "snapshot",
            runtime_checkpoint=runtime_checkpoint,
            include_actions=include_actions,
        )

    def restore(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        return self.call("restore", snapshot=snapshot)

    def checkpoint_metrics(self) -> dict[str, Any]:
        return self.call("checkpoint_metrics")

    def restore_checkpoint(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        return self.call("restore_checkpoint", snapshot=snapshot)

    def clear_runtime_checkpoints(self) -> dict[str, Any]:
        return self.call("clear_runtime_checkpoints")

    def pin_runtime_checkpoint(self, snapshot: dict[str, Any], pinned: bool = True) -> dict[str, Any]:
        return self.call("pin_runtime_checkpoint", snapshot=snapshot, pinned=bool(pinned))

    def redeterminize(self, side: str, seed: int) -> dict[str, Any]:
        return self.call("redeterminize", side=side, seed=seed)

    def observe(self, side: str) -> dict[str, Any]:
        return self.call("observe", side=side)

    def render_text(self, side: str | None = None) -> str:
        payload = {} if side is None else {"side": side}
        return self.call("render_text", **payload)

    def cards(self) -> list[dict[str, Any]]:
        return self.call("cards")

    def countries(self) -> list[dict[str, Any]]:
        return self.call("countries")

    def preset(self) -> dict[str, Any]:
        return self.call("preset")

    def log(self) -> dict[str, Any]:
        return self.call("log")
