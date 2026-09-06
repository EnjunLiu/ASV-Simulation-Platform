"""Call the official Unreal MCP server and wait for asset-pipeline results.

The server lives inside a running Unreal Editor at http://127.0.0.1:8000/mcp.
This module is used by Kit and by local CLI helpers. It never claims success
from a fire-and-forget process start.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_MCP_URL = "http://127.0.0.1:8000/mcp"
PROTOCOL_VERSION = "2025-11-25"


class UnrealMcpError(RuntimeError):
    pass


class UnrealMcpClient:
    def __init__(self, url: str = DEFAULT_MCP_URL, timeout: float = 120.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._id = 0
        self._session_id: str | None = None
        self._initialized = False
        self._protocol_version = PROTOCOL_VERSION

    def available(self) -> bool:
        host, port = self._host_port()
        sock = socket.socket()
        sock.settimeout(0.4)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def _host_port(self) -> tuple[str, int]:
        from urllib.parse import urlparse

        parsed = urlparse(self.url)
        return parsed.hostname or "127.0.0.1", parsed.port or 80

    def initialize(self) -> dict:
        result = self._rpc(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "kit-asset-pipeline", "version": "1.0"},
            },
            include_protocol_header=False,
        )
        negotiated = (result or {}).get("protocolVersion")
        if negotiated:
            self._protocol_version = negotiated
        self._rpc("notifications/initialized", notify=True)
        self._initialized = True
        return result

    def list_tools(self) -> list[dict]:
        self._ensure()
        result = self._rpc("tools/list") or {}
        return list(result.get("tools") or [])

    def call_tool(self, name: str, arguments: dict | None = None, timeout: float | None = None) -> Any:
        self._ensure()
        result = self._rpc(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
            timeout=timeout or self.timeout,
        )
        if isinstance(result, dict) and result.get("isError"):
            raise UnrealMcpError(_tool_text(result) or f"MCP tool error: {name}")
        return result

    def import_usd_manifest(self, manifest_path: str, timeout: float = 600.0) -> dict:
        self._ensure()
        last_error = None
        for toolset in ("asset_pipeline_toolset.AssetPipelineTools", "AssetPipelineTools"):
            try:
                payload = self.call_tool(
                    "call_tool",
                    {
                        "toolset_name": toolset,
                        "tool_name": "import_usd_manifest",
                        "arguments": {"manifest_path": manifest_path},
                    },
                    timeout=timeout,
                )
                parsed = _as_dict(payload)
                if parsed.get("status") == "imported" or "ue_target_mesh" in parsed:
                    return parsed
                last_error = parsed
            except UnrealMcpError as exc:
                last_error = exc
        raise UnrealMcpError(
            "Unreal MCP is up but AssetPipelineTools.import_usd_manifest is not registered. "
            f"Last error: {last_error}"
        )

    def asset_exists(self, object_path: str) -> bool:
        self._ensure()
        result = self.call_tool(
            "call_tool",
            {
                "toolset_name": "editor_toolset.toolsets.asset.AssetTools",
                "tool_name": "exists",
                "arguments": {"path": object_path},
            },
        )
        parsed = _as_dict(result)
        if "returnValue" in parsed:
            return bool(parsed["returnValue"])
        text = _tool_text(result).strip().lower()
        return text.endswith("true") or '"returnvalue": true' in text.replace(" ", "")

    def _ensure(self) -> None:
        if not self._initialized:
            self.initialize()

    def _rpc(
        self,
        method: str,
        params: dict | None = None,
        notify: bool = False,
        timeout: float | None = None,
        include_protocol_header: bool = True,
    ) -> Any:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if not notify:
            self._id += 1
            payload["id"] = self._id
        if params is not None:
            payload["params"] = params
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if include_protocol_header:
            headers["MCP-Protocol-Version"] = self._protocol_version
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        request = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                session = response.headers.get("Mcp-Session-Id")
                if session:
                    self._session_id = session
                parsed = _read_mcp_body(response)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise UnrealMcpError(f"MCP HTTP {exc.code}: {body[:800]}") from exc
        except urllib.error.URLError as exc:
            raise UnrealMcpError(f"Unreal MCP is not reachable at {self.url}: {exc}") from exc
        if notify:
            return None
        if not isinstance(parsed, dict):
            raise UnrealMcpError(f"Unexpected MCP payload: {parsed!r}")
        if parsed.get("error"):
            raise UnrealMcpError(str(parsed["error"]))
        return parsed.get("result")


def _read_mcp_body(response) -> Any:
    raw = response.read().decode("utf-8")
    content_type = response.headers.get("Content-Type", "")
    if "text/event-stream" in content_type:
        last = None
        for line in raw.splitlines():
            if line.startswith("data:"):
                chunk = line[5:].strip()
                if chunk and chunk != "[DONE]":
                    last = json.loads(chunk)
        if last is None:
            raise UnrealMcpError(f"Empty MCP SSE body: {raw[:400]}")
        return last
    if not raw.strip():
        return {}
    return json.loads(raw)


def _tool_text(result: Any) -> str:
    if isinstance(result, dict):
        parts = []
        for item in result.get("content") or []:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        if parts:
            return "\n".join(parts)
        return json.dumps(result, ensure_ascii=False)
    return str(result)


def _as_dict(result: Any) -> dict:
    text = _tool_text(result)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    if isinstance(result, dict) and "status" in result:
        return result
    return {"raw": text, "status": "unknown"}


def wait_for_mcp(url: str = DEFAULT_MCP_URL, timeout: float = 30.0) -> UnrealMcpClient:
    client = UnrealMcpClient(url)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.available():
            try:
                client.initialize()
                return client
            except UnrealMcpError:
                time.sleep(0.5)
                continue
        time.sleep(0.5)
    raise UnrealMcpError(f"Unreal MCP did not become ready at {url} within {timeout:.0f}s")
