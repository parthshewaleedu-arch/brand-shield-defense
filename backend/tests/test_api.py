from __future__ import annotations

import asyncio
import base64
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock

from backend.app.main import (
    app,
    get_classifier,
    get_feature_extractor,
    get_fusion,
    get_history_store,
    get_sandbox,
    get_sandbox_semaphore,
    get_ttl_cache,
    get_unwrapper,
    get_visual_matcher,
    SimpleTtlCache,
)
from backend.app.fusion import MultiModalRiskFusion
from backend.app.schemas import (
    BrandRecord,
    DomSignals,
    LexicalMlResult,
    RedirectChainResult,
    RedirectHop,
    SandboxResult,
    ShapReason,
    VisualCvResult,
    VisualMatchCandidate,
)
from backend.app.storage import ScanHistoryStore


@pytest.fixture
def mock_unwrapper():
    unwrapper = MagicMock()
    unwrapper.normalize_url = lambda url: url.strip()

    async def fake_unwrap(url: str, max_hops: int = 10):
        return RedirectChainResult(
            original_url=url,
            final_url=url,
            registered_domain="example.com",
            hops=[
                RedirectHop(
                    hop_number=1,
                    url=url,
                    status_code=200,
                    mechanism="final",
                    response_headers={},
                )
            ],
            total_hops=1,
            has_shortener=False,
            redirect_loop_detected=False,
            errors=[],
        )

    unwrapper.unwrap = AsyncMock(side_effect=fake_unwrap)
    return unwrapper


@pytest.fixture
def mock_sandbox():
    sandbox = MagicMock()

    async def fake_render(url: str):
        dom = DomSignals(
            page_title="Example Page",
            final_rendered_url=url,
            has_password_input=False,
            password_input_count=0,
            form_count=0,
            external_form_action_count=0,
            suspicious_form_actions=[],
            iframe_count=0,
            hidden_iframe_count=0,
            script_count=1,
            external_script_ratio=0.0,
            favicon_url=None,
            inline_event_handler_count=0,
            obfuscated_js_heuristic=False,
        )
        fake_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfakeimagebytes").decode("ascii")
        return SandboxResult(
            success=True,
            final_url=url,
            http_status=200,
            dom_signals=dom,
            screenshot_bytes_b64=fake_b64,
            screenshot_sha256="fake_sha256",
            render_time_ms=50,
            error=None,
        )

    sandbox.render = AsyncMock(side_effect=fake_render)
    sandbox.close = AsyncMock()
    return sandbox


@pytest.fixture
def mock_feature_extractor():
    extractor = MagicMock()

    async def fake_extract(url, redirect=None, dom=None):
        return {"url_length": len(url), "domain_age_days": 1000}, False, False

    extractor.extract = AsyncMock(side_effect=fake_extract)
    return extractor


@pytest.fixture
def mock_classifier():
    classifier = MagicMock()
    classifier.model = "mock_model"
    classifier.predict = MagicMock(
        return_value=LexicalMlResult(
            phishing_probability=0.04,
            risk_label="LOW",
            extracted_features={"url_length": 19},
            top_reasons=[
                ShapReason(
                    feature_name="domain_age_days",
                    feature_value=1000,
                    shap_impact=-0.25,
                    direction="decreases_risk",
                    explanation="Domain registered long ago",
                )
            ],
            whois_lookup_failed=False,
            ssl_lookup_failed=False,
            model_version="tabular_v1.0.0",
        )
    )
    return classifier


@pytest.fixture
def mock_visual_matcher():
    matcher = MagicMock()
    matcher.embedder = MagicMock(device="cpu")
    matcher.brand_meta = {
        "sbi": BrandRecord(
            brand_id="sbi",
            brand_name="State Bank of India",
            official_domains=["onlinesbi.sbi", "sbi.co.in"],
            category="banking",
            reference_images=["sbi_login.png"],
        )
    }

    def fake_match(target_url, image=None, image_bytes=None):
        return VisualCvResult(
            matched_brand_id="sbi",
            matched_brand_name="State Bank of India",
            official_domains=["onlinesbi.sbi", "sbi.co.in"],
            cosine_similarity=0.15,
            phash_distance=40,
            combined_visual_score=0.18,
            domain_is_official=False,
            impersonation_detected=False,
            top_candidates=[],
            inference_time_ms=5,
            skipped_reason=None,
        )

    matcher.match = MagicMock(side_effect=fake_match)
    return matcher


@pytest.fixture
async def api_client(
    tmp_path,
    mock_unwrapper,
    mock_sandbox,
    mock_feature_extractor,
    mock_classifier,
    mock_visual_matcher,
):
    """Async HTTP client with lightweight mock dependency overrides."""
    temp_db = tmp_path / "test_api_history.db"
    history_store = ScanHistoryStore(db_path=temp_db)
    ttl_cache = SimpleTtlCache(ttl_seconds=300.0)

    app.dependency_overrides[get_unwrapper] = lambda: mock_unwrapper
    app.dependency_overrides[get_sandbox] = lambda: mock_sandbox
    app.dependency_overrides[get_feature_extractor] = lambda: mock_feature_extractor
    app.dependency_overrides[get_classifier] = lambda: mock_classifier
    app.dependency_overrides[get_visual_matcher] = lambda: mock_visual_matcher
    app.dependency_overrides[get_fusion] = lambda: MultiModalRiskFusion()
    app.dependency_overrides[get_history_store] = lambda: history_store
    app.dependency_overrides[get_sandbox_semaphore] = lambda: asyncio.Semaphore(3)
    app.dependency_overrides[get_ttl_cache] = lambda: ttl_cache

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health_endpoint(api_client):
    """GET /api/v1/health returns 200 OK and model readiness."""
    res = await api_client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["tabular_model_loaded"] is True
    assert data["brand_vault_loaded"] is True
    assert data["indexed_brands_count"] == 1
    assert data["device"] == "cpu"
    assert data["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_brands_endpoint(api_client):
    """GET /api/v1/brands returns list of protected brands."""
    res = await api_client.get("/api/v1/brands")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["brand_id"] == "sbi"
    assert "onlinesbi.sbi" in data[0]["official_domains"]


@pytest.mark.asyncio
async def test_analyze_clean_site(api_client):
    """POST /api/v1/analyze evaluates clean website as SAFE."""
    payload = {"url": "https://example.com"}
    res = await api_client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["original_url"] == "https://example.com"
    assert data["fusion"]["verdict"] == "SAFE"
    assert data["fusion"]["recommended_action"] == "ALLOW"
    assert data["cached"] is False
    # include_screenshot default is False -> sandbox screenshot_bytes_b64 must be None
    assert data["sandbox"]["screenshot_bytes_b64"] is None


@pytest.mark.asyncio
async def test_analyze_include_screenshot(api_client):
    """POST /api/v1/analyze with include_screenshot=True returns base64 image."""
    payload = {"url": "https://example.com/login", "include_screenshot": True}
    res = await api_client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["sandbox"]["screenshot_bytes_b64"] is not None
    assert "fakeimagebytes" in base64.b64decode(data["sandbox"]["screenshot_bytes_b64"]).decode("latin1")


@pytest.mark.asyncio
async def test_analyze_visual_impersonation_override(api_client, mock_visual_matcher):
    """Visual brand impersonation triggers hard override to CRITICAL_PHISHING."""
    def fake_impersonation(target_url, image=None, image_bytes=None):
        return VisualCvResult(
            matched_brand_id="sbi",
            matched_brand_name="State Bank of India",
            official_domains=["onlinesbi.sbi", "sbi.co.in"],
            cosine_similarity=0.92,
            phash_distance=1,
            combined_visual_score=0.94,
            domain_is_official=False,
            impersonation_detected=True,
            top_candidates=[
                VisualMatchCandidate(
                    brand_id="sbi",
                    brand_name="State Bank of India",
                    cosine_similarity=0.92,
                    phash_distance=1,
                    phash_similarity=0.98,
                    combined_visual_score=0.94,
                )
            ],
            inference_time_ms=10,
            skipped_reason=None,
        )

    mock_visual_matcher.match = MagicMock(side_effect=fake_impersonation)

    payload = {"url": "https://sbi-login.evil-cloud.vercel.app"}
    res = await api_client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["fusion"]["verdict"] == "CRITICAL_PHISHING"
    assert data["fusion"]["recommended_action"] == "BLOCK_CREDENTIALS"
    assert data["fusion"]["override_triggered"] is True
    assert "State Bank of India" in data["fusion"]["override_reason"]
    assert data["fusion"]["final_risk_score"] >= 0.92


@pytest.mark.asyncio
async def test_invalid_url_schemes_rejected(api_client):
    """Dangerous URL schemes (javascript:, file:) return HTTP 422."""
    res1 = await api_client.post("/api/v1/analyze", json={"url": "javascript:alert(1)"})
    assert res1.status_code == 422

    res2 = await api_client.post("/api/v1/analyze", json={"url": "file:///etc/passwd"})
    assert res2.status_code == 422


@pytest.mark.asyncio
async def test_ttl_cache_hit(api_client):
    """Subsequent identical request hits in-memory TTL cache."""
    payload = {"url": "https://cache-test-domain.org"}
    res1 = await api_client.post("/api/v1/analyze", json=payload)
    assert res1.status_code == 200
    assert res1.json()["cached"] is False

    res2 = await api_client.post("/api/v1/analyze", json=payload)
    assert res2.status_code == 200
    assert res2.json()["cached"] is True
    assert res2.json()["original_url"] == "https://cache-test-domain.org"


@pytest.mark.asyncio
async def test_scan_history_recording_and_retrieval(api_client):
    """Analyzed requests are recorded in SQLite and retrievable via GET /api/v1/history."""
    await api_client.post("/api/v1/analyze", json={"url": "https://history-target-1.com"})
    await api_client.post("/api/v1/analyze", json={"url": "https://history-target-2.com"})

    res = await api_client.get("/api/v1/history?limit=10")
    assert res.status_code == 200
    items = res.json()
    assert len(items) >= 2
    urls = [it["original_url"] for it in items]
    assert "https://history-target-2.com" in urls
    assert "https://history-target-1.com" in urls


@pytest.mark.asyncio
async def test_cors_preflight_for_browser_extension(api_client):
    """OPTIONS preflight from Chrome extension returns 200 with CORS headers."""
    headers = {
        "Origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }
    res = await api_client.options("/api/v1/analyze", headers=headers)
    assert res.status_code == 200
    assert (
        res.headers.get("access-control-allow-origin")
        == "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
    )
    assert "POST" in res.headers.get("access-control-allow-methods", "")
