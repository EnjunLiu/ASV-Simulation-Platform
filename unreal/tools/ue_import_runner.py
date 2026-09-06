"""Run USD import into UE 5.8 and wait for a real result.

Prefers the official Unreal MCP server in an already-open editor. Falls back
to UnrealEditor-Cmd only when no editor is holding the project lock.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

from ue_mcp_client import DEFAULT_MCP_URL, UnrealMcpClient, UnrealMcpError

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = os.environ.get("UE_PROJECT") or str(REPO_ROOT / "HILPlatform.uproject")
DEFAULT_EDITOR_CMD = os.environ.get("UE_EDITOR_CMD") or os.environ.get(
    "UNREAL_EDITOR_CMD",
    "",
)
DEFAULT_SCRIPT = Path(__file__).resolve().parent / "import_usd_manifest.py"


def editor_process_running() -> bool:
    if os.name != "nt":
        return False
    try:
        completed = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq UnrealEditor.exe", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return "UnrealEditor.exe" in (completed.stdout or "")


def mcp_port_open(url: str = DEFAULT_MCP_URL) -> bool:
    return UnrealMcpClient(url).available()


def _load_result(manifest_path: Path, older_than: float) -> dict | None:
    report = manifest_path.with_name("ue_import_result.json")
    if not report.is_file():
        return None
    if report.stat().st_mtime < older_than:
        return None
    payload = json.loads(report.read_text(encoding="utf-8"))
    if payload.get("status") != "imported":
        raise RuntimeError(f"UE import reported failure: {payload}")
    return payload


def import_via_mcp(manifest_path: Path, timeout: float = 600.0) -> dict:
    started = time.time()
    client = UnrealMcpClient(timeout=timeout)
    if not client.available():
        raise UnrealMcpError("Unreal MCP port is closed")
    payload = client.import_usd_manifest(str(manifest_path), timeout=timeout)
    result = _load_result(manifest_path, started - 1.0)
    if result:
        return result
    if payload.get("status") == "imported":
        return payload
    raise RuntimeError(f"MCP import finished without ue_import_result.json: {payload}")


def import_via_commandlet(
    manifest_path: Path,
    editor_cmd: str,
    project: str,
    script: Path | None = None,
    timeout: float = 900.0,
) -> dict:
    if not editor_cmd:
        raise RuntimeError(
            "Set UE_EDITOR_CMD to UnrealEditor-Cmd.exe, or keep Unreal Editor open so MCP import can run."
        )
    if editor_process_running():
        raise RuntimeError(
            "Unreal Editor is already open; refusing UnrealEditor-Cmd so we do not fight the content lock. "
            "Use MCP import, or close the editor first."
        )
    script = script or DEFAULT_SCRIPT
    started = time.time()
    env = os.environ.copy()
    env["ASSET_PIPELINE_MANIFEST"] = str(manifest_path)
    command = [
        editor_cmd,
        project,
        "-unattended",
        "-nop4",
        "-nosplash",
        f"-ExecutePythonScript={script}",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        check=False,
    )
    result = _load_result(manifest_path, started - 1.0)
    if result:
        return result
    tail = ((completed.stderr or "") + "\n" + (completed.stdout or ""))[-1200:]
    raise RuntimeError(
        f"UnrealEditor-Cmd import failed (exit {completed.returncode}). {tail}"
    )


def import_and_wait(
    manifest_path: Path,
    editor_cmd: str = DEFAULT_EDITOR_CMD,
    project: str = DEFAULT_PROJECT,
    script: Path | None = None,
    timeout: float = 900.0,
) -> dict:
    manifest_path = Path(manifest_path).resolve()
    if mcp_port_open():
        return import_via_mcp(manifest_path, timeout=min(timeout, 600.0))
    return import_via_commandlet(manifest_path, editor_cmd, project, script=script, timeout=timeout)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--editor-cmd", default=DEFAULT_EDITOR_CMD)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    args = parser.parse_args()
    print(json.dumps(import_and_wait(args.manifest, args.editor_cmd, args.project), indent=2))
