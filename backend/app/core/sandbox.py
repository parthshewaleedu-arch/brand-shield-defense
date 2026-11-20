from __future__ import annotations

import base64
import hashlib
import ipaddress
import logging
import os
import shutil
import time
import urllib.parse
from pathlib import Path
from typing import Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Playwright,
    Response,
    async_playwright,
)

from backend.app.schemas import DomSignals, SandboxResult

logger = logging.getLogger("brandshield.sandbox")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

DOM_EXTRACTION_SCRIPT = """
() => {
    // 1. Document title
    const pageTitle = document.title ? document.title.trim() : null;

    // 2. Password inputs
    const passwordInputs = Array.from(document.querySelectorAll('input[type="password"]'));
    const passwordInputCount = passwordInputs.length;
    const hasPasswordInput = passwordInputCount > 0;

    // 3. Forms & actions
    const forms = Array.from(document.querySelectorAll('form'));
    const formCount = forms.length;
    const currentHost = window.location.hostname.toLowerCase();
    const suspiciousFormActions = [];
    let externalFormActionCount = 0;

    for (const form of forms) {
        let rawAction = form.getAttribute('action');
        if (!rawAction) {
            rawAction = "";
        }
        const trimmed = rawAction.trim();
        if (trimmed) {
            try {
                const actionUrl = new URL(trimmed, window.location.href);
                if (actionUrl.hostname && actionUrl.hostname.toLowerCase() !== currentHost) {
                    externalFormActionCount++;
                    suspiciousFormActions.push(actionUrl.href);
                }
            } catch (e) {
                // Malformed or dangerous actions like javascript: or data:
                if (trimmed.toLowerCase().startsWith('javascript:') || trimmed.toLowerCase().startsWith('data:')) {
                    suspiciousFormActions.push(trimmed);
                }
            }
        }
    }

    // 4. Iframes & hidden iframes
    const iframes = Array.from(document.querySelectorAll('iframe'));
    const iframeCount = iframes.length;
    let hiddenIframeCount = 0;

    for (const iframe of iframes) {
        let isHidden = false;
        try {
            const style = window.getComputedStyle(iframe);
            const rect = iframe.getBoundingClientRect();
            isHidden = (
                style.display === 'none' ||
                style.visibility === 'hidden' ||
                parseFloat(style.opacity || '1') === 0 ||
                rect.width <= 1 ||
                rect.height <= 1 ||
                iframe.hasAttribute('hidden') ||
                iframe.getAttribute('width') === '0' ||
                iframe.getAttribute('height') === '0'
            );
        } catch (e) {
            isHidden = false;
        }
        if (isHidden) {
            hiddenIframeCount++;
        }
    }

    // 5. Scripts & external script ratio & obfuscation heuristic
    const scripts = Array.from(document.querySelectorAll('script'));
    const scriptCount = scripts.length;
    let externalScriptCount = 0;
    let obfuscatedJsHeuristic = false;

    for (const script of scripts) {
        const src = script.getAttribute('src');
        if (src) {
            try {
                const scriptUrl = new URL(src, window.location.href);
                if (scriptUrl.hostname && scriptUrl.hostname.toLowerCase() !== currentHost) {
                    externalScriptCount++;
                }
            } catch (e) {}
        } else {
            const content = script.textContent || "";
            // Heuristics for packed/obfuscated JS patterns
            if (
                /\\beval\\s*\\(/i.test(content) &&
                (/\\bunescape\\s*\\(/i.test(content) || /\\batob\\s*\\(/i.test(content) || /\\bString\\.fromCharCode\\s*\\(/i.test(content))
            ) {
                obfuscatedJsHeuristic = true;
            } else if (/\\bdocument\\.write\\s*\\(\\s*unescape/i.test(content)) {
                obfuscatedJsHeuristic = true;
            } else if (/var\\s+[a-zA-Z0-9_$]+\\s*=\\s*\\[\\s*(?:["'][^"']+["']\\s*,\\s*){8,}\\]/i.test(content)) {
                obfuscatedJsHeuristic = true;
            }
        }
    }
    const externalScriptRatio = scriptCount > 0 ? (externalScriptCount / scriptCount) : 0.0;

    // 6. Favicon URL
    let faviconUrl = null;
    const faviconLink = document.querySelector('link[rel~="icon"]') || document.querySelector('link[rel="shortcut icon"]');
    if (faviconLink && faviconLink.getAttribute('href')) {
        try {
            faviconUrl = new URL(faviconLink.getAttribute('href'), window.location.href).href;
        } catch (e) {
            faviconUrl = faviconLink.getAttribute('href');
        }
    }

    // 7. Inline event handlers (onclick, onload, etc.)
    let inlineEventHandlerCount = 0;
    const eventHandlerRegex = /^on[a-z]+/i;
    const allElements = document.querySelectorAll('*');
    for (const el of allElements) {
        for (const attr of el.attributes) {
            if (eventHandlerRegex.test(attr.name)) {
                inlineEventHandlerCount++;
            }
        }
    }

    return {
        page_title: pageTitle,
        final_rendered_url: window.location.href,
        has_password_input: hasPasswordInput,
        password_input_count: passwordInputCount,
        form_count: formCount,
        external_form_action_count: externalFormActionCount,
        suspicious_form_actions: suspiciousFormActions,
        iframe_count: iframeCount,
        hidden_iframe_count: hiddenIframeCount,
        script_count: scriptCount,
        external_script_ratio: Math.round(externalScriptRatio * 1000) / 1000,
        favicon_url: faviconUrl,
        inline_event_handler_count: inlineEventHandlerCount,
        obfuscated_js_heuristic: obfuscatedJsHeuristic
    };
}
"""


def detect_chromium_executable() -> Optional[str]:
    """Locate local Chromium executable binary across Playwright cache and system paths."""
    env_path = os.environ.get("PLAYWRIGHT_CHROME_PATH") or os.environ.get("CHROME_PATH")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path

    cache_dir = Path.home() / ".cache" / "ms-playwright"
    if cache_dir.exists():
        for chrome_bin in cache_dir.glob("chromium-*/chrome-linux64/chrome"):
            if chrome_bin.is_file() and os.access(chrome_bin, os.X_OK):
                return str(chrome_bin)
        for shell_bin in cache_dir.glob("chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"):
            if shell_bin.is_file() and os.access(shell_bin, os.X_OK):
                return str(shell_bin)

    for sys_cmd in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        found = shutil.which(sys_cmd)
        if found:
            return found

    return None


class BrowserSandbox:
    """Hardened, sandboxed headless Chromium browser service for safe page rendering and DOM telemetry extraction."""

    def __init__(
        self,
        navigation_timeout_ms: int = 8000,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        user_agent: Optional[str] = None,
        allow_localhost: bool = True,
        block_media_streams: bool = True,
    ) -> None:
        self.navigation_timeout_ms = navigation_timeout_ms
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self.allow_localhost = allow_localhost
        self.block_media_streams = block_media_streams

        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._executable_path = detect_chromium_executable()

    async def start(self) -> None:
        """Initialize playwright and launch browser if not already started."""
        if self._browser is not None:
            return

        self._playwright = await async_playwright().start()

        launch_kwargs = {
            "headless": True,
            "args": [
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-setuid-sandbox",
            ],
        }
        if self._executable_path:
            launch_kwargs["executable_path"] = self._executable_path

        try:
            self._browser = await self._playwright.chromium.launch(**launch_kwargs)
        except Exception as exc:
            logger.warning("Default launch failed (%s); trying fallback launch without custom executable", exc)
            launch_kwargs.pop("executable_path", None)
            self._browser = await self._playwright.chromium.launch(**launch_kwargs)

        logger.info("Playwright sandbox browser launched successfully")

    async def close(self) -> None:
        """Close browser instance and terminate playwright to prevent zombie processes."""
        if self._browser:
            try:
                await self._browser.close()
            except Exception as exc:
                logger.debug("Error closing browser: %s", exc)
            finally:
                self._browser = None

        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception as exc:
                logger.debug("Error stopping playwright: %s", exc)
            finally:
                self._playwright = None

        logger.info("Playwright sandbox resources closed cleanly")

    async def __aenter__(self) -> BrowserSandbox:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def _is_ssrf_risk(self, url: str) -> bool:
        """Check if target host is a private/loopback address when localhost is disallowed."""
        if self.allow_localhost:
            return False
        try:
            parsed = urllib.parse.urlparse(url)
            hostname = (parsed.hostname or "").strip().lower()
            if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
                return True
            ip = ipaddress.ip_address(hostname)
            return ip.is_private or ip.is_loopback or ip.is_link_local
        except ValueError:
            return False

    async def render(self, url: str) -> SandboxResult:
        """
        Safely render the specified URL, enforce sandbox policies,
        extract DomSignals, and capture a 1280x720 PNG screenshot buffer.
        """
        start_time = time.perf_counter()
        normalized_url = (url or "").strip()

        # Guard against SSRF if enabled
        if self._is_ssrf_risk(normalized_url):
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return SandboxResult(
                success=False,
                final_url=normalized_url,
                http_status=None,
                dom_signals=DomSignals(final_rendered_url=normalized_url),
                screenshot_bytes_b64=None,
                screenshot_sha256=None,
                render_time_ms=elapsed_ms,
                error="Access to private or loopback IP address blocked by sandbox SSRF guard",
            )

        # Ensure browser is started
        if self._browser is None:
            await self.start()

        assert self._browser is not None

        context: Optional[BrowserContext] = None
        page = None
        http_status: Optional[int] = None

        try:
            # Create isolated context with mandatory security constraints
            context = await self._browser.new_context(
                viewport={"width": self.viewport_width, "height": self.viewport_height},
                device_scale_factor=1,
                user_agent=self.user_agent,
                accept_downloads=False,
                ignore_https_errors=True,
                java_script_enabled=True,
            )
            context.set_default_navigation_timeout(self.navigation_timeout_ms)

            # Block heavy streaming media if configured, while retaining stylesheets, scripts, and images
            if self.block_media_streams:
                async def route_handler(route):
                    req = route.request
                    if req.resource_type in ("media",):
                        await route.abort()
                    else:
                        await route.continue_()

                await context.route("**/*", route_handler)

            page = await context.new_page()

            # Record main document response status
            def on_response(response: Response):
                nonlocal http_status
                if response.url == page.url or http_status is None:
                    http_status = response.status

            page.on("response", on_response)

            # Navigate to target with 8000ms navigation timeout
            try:
                response = await page.goto(
                    normalized_url,
                    timeout=self.navigation_timeout_ms,
                    wait_until="domcontentloaded",
                )
                if response:
                    http_status = response.status
            except Exception as nav_exc:
                # If domcontentloaded timed out or failed, attempt to inspect whatever DOM rendered
                logger.warning("Navigation warning for %s: %s", normalized_url, nav_exc)
                if http_status is None:
                    # Reraise to outer handler if complete failure
                    raise nav_exc

            # Settle network/DOM briefly (up to 300ms)
            try:
                await page.wait_for_load_state("load", timeout=1000)
            except Exception:
                pass

            # Extract DOM telemetry via evaluation script
            raw_dom = await page.evaluate(DOM_EXTRACTION_SCRIPT)
            dom_signals = DomSignals(**raw_dom)

            # Capture 1280x720 viewport PNG screenshot
            screenshot_bytes = await page.screenshot(type="png", full_page=False)
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            screenshot_sha256 = hashlib.sha256(screenshot_bytes).hexdigest()

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            return SandboxResult(
                success=True,
                final_url=dom_signals.final_rendered_url or normalized_url,
                http_status=http_status,
                dom_signals=dom_signals,
                screenshot_bytes_b64=screenshot_b64,
                screenshot_sha256=screenshot_sha256,
                render_time_ms=elapsed_ms,
                error=None,
            )

        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            error_message = f"Sandbox rendering failed: {type(exc).__name__}: {str(exc)}"
            logger.error(error_message)

            fallback_signals = DomSignals(
                page_title=None,
                final_rendered_url=normalized_url,
                has_password_input=False,
                password_input_count=0,
                form_count=0,
                external_form_action_count=0,
                suspicious_form_actions=[],
                iframe_count=0,
                hidden_iframe_count=0,
                script_count=0,
                external_script_ratio=0.0,
                favicon_url=None,
                inline_event_handler_count=0,
                obfuscated_js_heuristic=False,
            )

            return SandboxResult(
                success=False,
                final_url=normalized_url,
                http_status=http_status,
                dom_signals=fallback_signals,
                screenshot_bytes_b64=None,
                screenshot_sha256=None,
                render_time_ms=elapsed_ms,
                error=error_message,
            )

        finally:
            # Guarantee cleanup of page and context to avoid memory/process leaks
            if page:
                try:
                    await page.close()
                except Exception:
                    pass
            if context:
                try:
                    await context.close()
                except Exception:
                    pass
