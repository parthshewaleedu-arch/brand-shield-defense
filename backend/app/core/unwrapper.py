from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Optional, Set
import httpx
import tldextract

from backend.app.schemas import RedirectChainResult, RedirectHop

logger = logging.getLogger("brandshield.unwrapper")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

KNOWN_SHORTENERS: set[str] = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "buff.ly",
    "is.gd", "rb.gy", "cutt.ly", "rebrand.ly", "shorturl.at", "lnkd.in",
    "tiny.cc", "bc.vc", "v.gd", "qr.ae", "adf.ly", "bl.ink", "trib.al",
    "cli.re", "snip.ly", "s.id", "soo.gd", "git.io", "ift.tt", "bitly.com",
    "clck.ru", "rotf.lol", "vzturl.com", "t.ly", "po.st"
}

SENSITIVE_HEADERS: set[str] = {
    "set-cookie", "cookie", "authorization", "proxy-authorization"
}

META_REFRESH_PATTERN = re.compile(
    r'<meta[^>]*?http-equiv\s*=\s*["\']?refresh["\']?[^>]*?content\s*=\s*["\']?\s*\d+\s*;\s*url\s*=\s*([^"\'\s>]+)',
    re.IGNORECASE
)
META_REFRESH_REVERSE_PATTERN = re.compile(
    r'<meta[^>]*?content\s*=\s*["\']?\s*\d+\s*;\s*url\s*=\s*([^"\'\s>]+)["\']?[^>]*?http-equiv\s*=\s*["\']?refresh["\']?',
    re.IGNORECASE
)

JS_REDIRECT_PATTERNS = [
    re.compile(r'(?:window\.|document\.)?location\.(?:replace|assign)\s*\(\s*["\']([^"\']+)["\']\s*\)', re.IGNORECASE),
    re.compile(r'(?:window\.|document\.)?location\.href\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'(?:window\.|document\.)?location\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE),
]


class RedirectUnwrapper:
    """Traces multi-hop redirect chains across HTTP 3xx, HTML meta-refresh, and JS location jumps."""

    def __init__(
        self,
        max_hops: int = 10,
        timeout_seconds: float = 6.0,
        user_agent: Optional[str] = None
    ) -> None:
        self.max_hops = max_hops
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent or DEFAULT_USER_AGENT

    @staticmethod
    def extract_registered_domain(url: str) -> str:
        """Safely extract the registered root domain (eTLD+1) with a fallback parser."""
        if not url:
            return ""
        try:
            ext = tldextract.extract(url)
            # Prefer top_domain_under_public_suffix (tldextract v5+) to avoid deprecation warnings
            if hasattr(ext, "top_domain_under_public_suffix"):
                domain_val = ext.top_domain_under_public_suffix
            else:
                domain_val = ext.registered_domain

            if domain_val:
                return domain_val.lower()
            if ext.domain:
                suffix = f".{ext.suffix}" if ext.suffix else ""
                return f"{ext.domain}{suffix}".lower()
        except Exception as exc:
            logger.debug("tldextract failed on %s: %s; using urlparse fallback", url, exc)

        try:
            parsed = urllib.parse.urlparse(url)
            host = (parsed.netloc or parsed.path).split(":")[0].strip().lower()
            parts = host.split(".")
            if len(parts) >= 2:
                return ".".join(parts[-2:])
            return host
        except Exception:
            return ""

    @classmethod
    def is_shortener(cls, url: str) -> bool:
        """Check if target URL domain matches known URL shorteners."""
        domain = cls.extract_registered_domain(url)
        if domain in KNOWN_SHORTENERS:
            return True
        try:
            parsed = urllib.parse.urlparse(url)
            host = (parsed.netloc or "").split(":")[0].lower()
            return host in KNOWN_SHORTENERS
        except Exception:
            return False

    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize URL for comparison and deduplication."""
        stripped = url.strip()
        parsed = urllib.parse.urlparse(stripped)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path or "/"
        # Strip trailing slash if path != '/'
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]
        normalized = urllib.parse.urlunparse((
            scheme,
            netloc,
            path,
            parsed.params,
            parsed.query,
            ""  # Strip fragment
        ))
        return normalized

    @staticmethod
    def sanitize_headers(headers: httpx.Headers) -> dict[str, str]:
        """Convert header keys to lowercase and redact sensitive headers."""
        sanitized: dict[str, str] = {}
        for k, v in headers.items():
            key_lower = k.lower()
            if key_lower in SENSITIVE_HEADERS:
                sanitized[key_lower] = "[REDACTED]"
            else:
                sanitized[key_lower] = v
        return sanitized

    @classmethod
    def find_html_redirect(cls, html_body: str, current_url: str) -> tuple[Optional[str], Optional[str]]:
        """
        Inspect HTML body for meta-refresh and JS location jumps.
        Returns (next_url, mechanism) or (None, None).
        """
        if not html_body or len(html_body) < 10:
            return None, None

        # 1. Meta refresh check
        match_meta = META_REFRESH_PATTERN.search(html_body)
        if not match_meta:
            match_meta = META_REFRESH_REVERSE_PATTERN.search(html_body)
        if match_meta:
            raw_target = match_meta.group(1).strip()
            # Strip quotes if captured
            raw_target = raw_target.strip("\"'")
            if raw_target:
                resolved = urllib.parse.urljoin(current_url, raw_target)
                return resolved, "meta_refresh"

        # 2. JavaScript location jumps
        for pattern in JS_REDIRECT_PATTERNS:
            match_js = pattern.search(html_body)
            if match_js:
                raw_target = match_js.group(1).strip()
                if raw_target and not raw_target.startswith("javascript:"):
                    resolved = urllib.parse.urljoin(current_url, raw_target)
                    return resolved, "js_redirect"

        return None, None

    async def unwrap(self, url: str, max_hops: Optional[int] = None) -> RedirectChainResult:
        """Trace the full redirect chain starting from the provided URL."""
        hops: list[RedirectHop] = []
        errors: list[str] = []
        seen_normalized_urls: Set[str] = set()
        has_shortener = False
        redirect_loop_detected = False

        effective_max_hops = max_hops if max_hops is not None else self.max_hops
        original_url = (url or "").strip()

        # Scheme validation
        if not re.match(r"^https?://", original_url, re.IGNORECASE):
            errors.append(f"Rejected invalid or dangerous URL scheme in '{original_url}'")
            return RedirectChainResult(
                original_url=original_url,
                final_url=original_url,
                registered_domain=self.extract_registered_domain(original_url),
                hops=[],
                total_hops=0,
                has_shortener=False,
                redirect_loop_detected=False,
                errors=errors,
            )

        current_url = original_url
        client_headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=self.timeout_seconds,
            verify=False,
        ) as client:
            while len(hops) < effective_max_hops:
                hop_number = len(hops) + 1
                normalized_current = self.normalize_url(current_url)

                if self.is_shortener(current_url):
                    has_shortener = True

                # Circular loop check
                if normalized_current in seen_normalized_urls:
                    redirect_loop_detected = True
                    error_msg = f"Redirect loop detected at hop {hop_number}: {current_url}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    break

                seen_normalized_urls.add(normalized_current)

                try:
                    response = await client.get(current_url, headers=client_headers)
                except httpx.TimeoutException as exc:
                    error_msg = f"HTTP request timeout at hop {hop_number} ({current_url}): {exc}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    hops.append(
                        RedirectHop(
                            hop_number=hop_number,
                            url=current_url,
                            status_code=None,
                            mechanism="final" if hop_number > 1 else "initial",
                            response_headers={},
                        )
                    )
                    break
                except httpx.RequestError as exc:
                    error_msg = f"HTTP request error at hop {hop_number} ({current_url}): {exc}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    hops.append(
                        RedirectHop(
                            hop_number=hop_number,
                            url=current_url,
                            status_code=None,
                            mechanism="final" if hop_number > 1 else "initial",
                            response_headers={},
                        )
                    )
                    break
                except Exception as exc:
                    error_msg = f"Unexpected error at hop {hop_number} ({current_url}): {exc}"
                    logger.error(error_msg, exc_info=True)
                    errors.append(error_msg)
                    hops.append(
                        RedirectHop(
                            hop_number=hop_number,
                            url=current_url,
                            status_code=None,
                            mechanism="final" if hop_number > 1 else "initial",
                            response_headers={},
                        )
                    )
                    break

                sanitized_headers = self.sanitize_headers(response.headers)
                status = response.status_code

                # 1. Check HTTP Redirects (301, 302, 303, 307, 308)
                if status in (301, 302, 303, 307, 308) and "location" in response.headers:
                    raw_location = response.headers["location"].strip()
                    next_url = urllib.parse.urljoin(current_url, raw_location)
                    hops.append(
                        RedirectHop(
                            hop_number=hop_number,
                            url=current_url,
                            status_code=status,
                            mechanism="http_redirect",
                            response_headers=sanitized_headers,
                        )
                    )
                    current_url = next_url
                    continue

                # 2. Check HTML Redirects (Meta Refresh / JS redirect) if 200 OK
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" in content_type or "application/xhtml" in content_type or status == 200:
                    html_text = response.text
                    next_url, redirect_mechanism = self.find_html_redirect(html_text, current_url)
                    if next_url and redirect_mechanism:
                        # Avoid immediate self-redirect
                        if self.normalize_url(next_url) != normalized_current:
                            hops.append(
                                RedirectHop(
                                    hop_number=hop_number,
                                    url=current_url,
                                    status_code=status,
                                    mechanism=redirect_mechanism,  # type: ignore[arg-type]
                                    response_headers=sanitized_headers,
                                )
                            )
                            current_url = next_url
                            continue

                # 3. No further redirects found: this is the final landing page
                mechanism_val = "initial" if hop_number == 1 else "final"
                hops.append(
                    RedirectHop(
                        hop_number=hop_number,
                        url=current_url,
                        status_code=status,
                        mechanism=mechanism_val,
                        response_headers=sanitized_headers,
                    )
                )
                break
            else:
                # Loop completed without break: max hops reached
                error_msg = f"Maximum hops cap ({self.max_hops}) reached before resolving terminal landing URL"
                logger.warning(error_msg)
                errors.append(error_msg)

        final_url = hops[-1].url if hops else current_url
        registered_domain = self.extract_registered_domain(final_url)

        return RedirectChainResult(
            original_url=original_url,
            final_url=final_url,
            registered_domain=registered_domain,
            hops=hops,
            total_hops=len(hops),
            has_shortener=has_shortener,
            redirect_loop_detected=redirect_loop_detected,
            errors=errors,
        )
