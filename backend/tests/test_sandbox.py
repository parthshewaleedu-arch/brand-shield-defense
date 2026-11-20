from __future__ import annotations

import base64
import pytest
from backend.app.core.sandbox import BrowserSandbox
from backend.app.schemas import SandboxResult


@pytest.mark.asyncio
async def test_sandbox_dom_signals_and_screenshot(mock_server: str) -> None:
    async with BrowserSandbox(allow_localhost=True, navigation_timeout_ms=8000) as sandbox:
        target_url = f"{mock_server}/sandbox-phish"
        result: SandboxResult = await sandbox.render(target_url)

        assert result.success is True
        assert result.error is None
        assert result.http_status == 200

        # Verify DOM telemetry signals
        dom = result.dom_signals
        assert dom.page_title == "SBI NetBanking Secure Login"
        assert dom.final_rendered_url.startswith(mock_server)
        assert dom.has_password_input is True
        assert dom.password_input_count == 1
        assert dom.form_count == 1
        assert dom.external_form_action_count == 1
        assert "https://phish-harvesting-server.xyz/collect" in dom.suspicious_form_actions
        assert dom.iframe_count == 2
        assert dom.hidden_iframe_count == 1
        assert dom.script_count == 2
        assert 0.4 <= dom.external_script_ratio <= 0.6
        assert dom.favicon_url is not None
        assert dom.favicon_url.endswith("/favicon.ico")
        assert dom.inline_event_handler_count >= 1

        # Verify Viewport Screenshot
        assert result.screenshot_bytes_b64 is not None
        raw_png = base64.b64decode(result.screenshot_bytes_b64)
        # PNG signature check: \x89PNG\r\n\x1a\n
        assert raw_png.startswith(b"\x89PNG\r\n\x1a\n")
        assert result.screenshot_sha256 is not None
        assert len(result.screenshot_sha256) == 64
        assert result.render_time_ms > 0


@pytest.mark.asyncio
async def test_sandbox_obfuscated_js_heuristic(mock_server: str) -> None:
    async with BrowserSandbox(allow_localhost=True, navigation_timeout_ms=8000) as sandbox:
        target_url = f"{mock_server}/sandbox-obfuscated"
        result: SandboxResult = await sandbox.render(target_url)

        assert result.success is True
        assert result.dom_signals.obfuscated_js_heuristic is True


@pytest.mark.asyncio
async def test_sandbox_unreachable_target() -> None:
    # Port 59997 has no listening service
    async with BrowserSandbox(allow_localhost=True, navigation_timeout_ms=2000) as sandbox:
        result: SandboxResult = await sandbox.render("http://127.0.0.1:59997/nonexistent")

        assert result.success is False
        assert result.error is not None
        assert result.screenshot_bytes_b64 is None
        assert result.dom_signals.has_password_input is False


@pytest.mark.asyncio
async def test_sandbox_ssrf_blocking() -> None:
    # Localhost disallowed in strict mode
    async with BrowserSandbox(allow_localhost=False, navigation_timeout_ms=2000) as sandbox:
        result: SandboxResult = await sandbox.render("http://127.0.0.1:8080/sensitive")

        assert result.success is False
        assert result.error is not None
        assert "ssrf" in result.error.lower() or "loopback" in result.error.lower()
