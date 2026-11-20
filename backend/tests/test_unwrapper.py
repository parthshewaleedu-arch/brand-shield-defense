from __future__ import annotations

import pytest
from backend.app.core.unwrapper import RedirectUnwrapper
from backend.app.schemas import RedirectChainResult


@pytest.mark.asyncio
async def test_normal_url_no_redirect(mock_server: str) -> None:
    unwrapper = RedirectUnwrapper(max_hops=10, timeout_seconds=4.0)
    target_url = f"{mock_server}/normal"

    result: RedirectChainResult = await unwrapper.unwrap(target_url)

    assert result.original_url == target_url
    assert result.final_url == target_url
    assert result.total_hops == 1
    assert len(result.hops) == 1
    assert result.hops[0].status_code == 200
    assert result.hops[0].mechanism in ("initial", "final")
    assert result.has_shortener is False
    assert result.redirect_loop_detected is False
    assert len(result.errors) == 0


@pytest.mark.asyncio
async def test_multi_hop_redirect_chain(mock_server: str) -> None:
    unwrapper = RedirectUnwrapper(max_hops=10, timeout_seconds=4.0)
    target_url = f"{mock_server}/hop1"

    result: RedirectChainResult = await unwrapper.unwrap(target_url)

    assert result.total_hops == 3
    assert len(result.hops) == 3
    assert result.final_url == f"{mock_server}/hop3"

    # Hop 1: 301 redirect
    assert result.hops[0].status_code == 301
    assert result.hops[0].mechanism == "http_redirect"
    assert result.hops[0].url == f"{mock_server}/hop1"

    # Hop 2: 302 redirect
    assert result.hops[1].status_code == 302
    assert result.hops[1].mechanism == "http_redirect"
    assert result.hops[1].url == f"{mock_server}/hop2"

    # Hop 3: 200 terminal landing
    assert result.hops[2].status_code == 200
    assert result.hops[2].mechanism == "final"
    assert result.hops[2].url == f"{mock_server}/hop3"
    assert result.redirect_loop_detected is False


@pytest.mark.asyncio
async def test_meta_refresh_redirect(mock_server: str) -> None:
    unwrapper = RedirectUnwrapper(max_hops=10, timeout_seconds=4.0)
    target_url = f"{mock_server}/meta-start"

    result: RedirectChainResult = await unwrapper.unwrap(target_url)

    assert result.total_hops == 2
    assert result.final_url == f"{mock_server}/meta-target"
    assert result.hops[0].mechanism == "meta_refresh"
    assert result.hops[1].mechanism == "final"
    assert result.hops[1].status_code == 200


@pytest.mark.asyncio
async def test_js_location_redirect(mock_server: str) -> None:
    unwrapper = RedirectUnwrapper(max_hops=10, timeout_seconds=4.0)
    target_url = f"{mock_server}/js-start"

    result: RedirectChainResult = await unwrapper.unwrap(target_url)

    assert result.total_hops == 2
    assert result.final_url == f"{mock_server}/js-target"
    assert result.hops[0].mechanism == "js_redirect"
    assert result.hops[1].mechanism == "final"
    assert result.hops[1].status_code == 200


@pytest.mark.asyncio
async def test_redirect_loop_detection(mock_server: str) -> None:
    unwrapper = RedirectUnwrapper(max_hops=10, timeout_seconds=4.0)
    target_url = f"{mock_server}/loop-a"

    result: RedirectChainResult = await unwrapper.unwrap(target_url)

    assert result.redirect_loop_detected is True
    assert any("loop detected" in err.lower() for err in result.errors)
    # Loop must terminate deterministically without exceeding max hops
    assert result.total_hops <= 3


@pytest.mark.asyncio
async def test_max_hops_truncation(mock_server: str) -> None:
    # Set tight hop cap of 3
    unwrapper = RedirectUnwrapper(max_hops=3, timeout_seconds=4.0)
    target_url = f"{mock_server}/chain/1"

    result: RedirectChainResult = await unwrapper.unwrap(target_url)

    assert result.total_hops == 3
    assert any("maximum hops" in err.lower() or "cap" in err.lower() for err in result.errors)


def test_shortener_detection() -> None:
    unwrapper = RedirectUnwrapper()

    assert unwrapper.is_shortener("https://bit.ly/3kPhish") is True
    assert unwrapper.is_shortener("https://tinyurl.com/bank-login") is True
    assert unwrapper.is_shortener("https://t.co/xyz123") is True
    assert unwrapper.is_shortener("https://is.gd/verify") is True
    assert unwrapper.is_shortener("https://cutt.ly/safe") is True
    assert unwrapper.is_shortener("https://google.com/search") is False
    assert unwrapper.is_shortener("https://onlinesbi.sbi/portal") is False


@pytest.mark.asyncio
async def test_invalid_scheme_rejection() -> None:
    unwrapper = RedirectUnwrapper()

    dangerous_urls = [
        "javascript:alert('xss')",
        "file:///etc/passwd",
        "data:text/html,<h1>phish</h1>",
        "ftp://mirror.example.com",
    ]

    for bad_url in dangerous_urls:
        result = await unwrapper.unwrap(bad_url)
        assert result.total_hops == 0
        assert len(result.hops) == 0
        assert any("invalid or dangerous" in err.lower() for err in result.errors)


@pytest.mark.asyncio
async def test_network_error_graceful() -> None:
    # Port 59998 has no listening service
    unwrapper = RedirectUnwrapper(timeout_seconds=1.5)
    result = await unwrapper.unwrap("http://127.0.0.1:59998/offline")

    assert len(result.errors) > 0
    assert result.hops[0].status_code is None
    assert result.total_hops == 1


def test_extract_registered_domain() -> None:
    unwrapper = RedirectUnwrapper()

    assert unwrapper.extract_registered_domain("https://login.sbi-verify.vercel.app/index.html") in ("vercel.app", "sbi-verify.vercel.app")
    assert unwrapper.extract_registered_domain("http://phishing.co.uk/account") == "phishing.co.uk"
    assert unwrapper.extract_registered_domain("https://secure.bank.com:8443/login") == "bank.com"
    assert unwrapper.extract_registered_domain("http://127.0.0.1:8000/test") == "127.0.0.1"
