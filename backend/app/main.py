from __future__ import annotations

import asyncio
import base64
import datetime
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.lexical_ml import LexicalPhishingClassifier, TabularFeatureExtractor
from .core.sandbox import BrowserSandbox
from .core.unwrapper import RedirectUnwrapper
from .core.visual_cv import BrandVisualMatcher
from .fusion import MultiModalRiskFusion
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    ScanHistoryItem,
)
from .storage import ScanHistoryStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("brandshield.api")


class SimpleTtlCache:
    """In-memory bounded TTL cache for analyzed URL responses."""

    def __init__(self, ttl_seconds: float = 300.0, max_entries: int = 256) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._cache: dict[tuple[str, bool], tuple[float, AnalyzeResponse]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: tuple[str, bool]) -> Optional[AnalyzeResponse]:
        async with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            ts, resp = entry
            if time.time() - ts > self.ttl_seconds:
                del self._cache[key]
                return None
            return resp

    async def set(self, key: tuple[str, bool], response: AnalyzeResponse) -> None:
        async with self._lock:
            if len(self._cache) >= self.max_entries:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (time.time(), response)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager initializing subsystems and cleaning up resources."""
    logger.info("Initializing BrandShield AI backend subsystems...")
    base_dir = Path(__file__).resolve().parents[1]

    # Initialize components if not already on state
    if not hasattr(app.state, "unwrapper"):
        app.state.unwrapper = RedirectUnwrapper()
    if not hasattr(app.state, "sandbox"):
        app.state.sandbox = BrowserSandbox()
    if not hasattr(app.state, "feature_extractor"):
        app.state.feature_extractor = TabularFeatureExtractor()
    if not hasattr(app.state, "classifier"):
        app.state.classifier = LexicalPhishingClassifier()

    # Load Tabular ML model artifact if present
    model_path = base_dir / "models" / "tabular_model.joblib"
    if model_path.exists() and app.state.classifier.model is None:
        try:
            app.state.classifier.load(model_path)
            logger.info("Loaded Tabular ML model from %s", model_path)
        except Exception as e:
            logger.warning("Could not load model at %s: %s", model_path, e)
    elif app.state.classifier.model is None:
        logger.warning("Model file not found at %s; tabular classifier running uninitialized", model_path)

    # Visual CV components
    if not hasattr(app.state, "visual_matcher"):
        vault_index = base_dir / "models" / "brand_vault_index.npz"
        vault_meta = base_dir / "models" / "brand_vault_meta.json"
        app.state.visual_matcher = BrandVisualMatcher(
            index_path=vault_index if vault_index.exists() else None,
            meta_path=vault_meta if vault_meta.exists() else None,
        )

    # Fusion and Storage
    if not hasattr(app.state, "fusion"):
        app.state.fusion = MultiModalRiskFusion()
    if not hasattr(app.state, "history_store"):
        app.state.history_store = ScanHistoryStore()

    # Bounded concurrency for headless Chromium sandbox
    if not hasattr(app.state, "sandbox_semaphore"):
        app.state.sandbox_semaphore = asyncio.Semaphore(3)

    # In-memory TTL cache (5-minute TTL, max 256 entries)
    if not hasattr(app.state, "ttl_cache"):
        app.state.ttl_cache = SimpleTtlCache(ttl_seconds=300.0, max_entries=256)

    logger.info("BrandShield AI backend initialization complete.")
    yield

    logger.info("Shutting down BrandShield AI backend resources...")
    if hasattr(app.state, "sandbox") and app.state.sandbox:
        await app.state.sandbox.close()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="BrandShield AI - Phishing & Visual Impersonation Defense API",
    description="Multi-Modal Brand Impersonation and Visual Phishing Defense System",
    version="1.0.0",
    lifespan=lifespan,
)

# Seed default state so test transports work without explicit lifespan trigger
app.state.unwrapper = RedirectUnwrapper()
app.state.sandbox = None
app.state.feature_extractor = TabularFeatureExtractor()
app.state.classifier = LexicalPhishingClassifier()
app.state.visual_matcher = BrandVisualMatcher()
app.state.fusion = MultiModalRiskFusion()
app.state.history_store = ScanHistoryStore()
app.state.sandbox_semaphore = asyncio.Semaphore(3)
app.state.ttl_cache = SimpleTtlCache(ttl_seconds=300.0, max_entries=256)


# CORS middleware for Chrome/Edge Manifest V3 extensions and local dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension://.*|moz-extension://.*|http://localhost(:\d+)?|http://127\.0\.0\.1(:\d+)?)$",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)


# Dependency injection helpers
def get_unwrapper(request: Request) -> RedirectUnwrapper:
    return request.app.state.unwrapper


def get_sandbox(request: Request) -> BrowserSandbox:
    return request.app.state.sandbox


def get_feature_extractor(request: Request) -> TabularFeatureExtractor:
    return request.app.state.feature_extractor


def get_classifier(request: Request) -> LexicalPhishingClassifier:
    return request.app.state.classifier


def get_visual_matcher(request: Request) -> BrandVisualMatcher:
    return request.app.state.visual_matcher


def get_fusion(request: Request) -> MultiModalRiskFusion:
    return request.app.state.fusion


def get_history_store(request: Request) -> ScanHistoryStore:
    return request.app.state.history_store


def get_sandbox_semaphore(request: Request) -> asyncio.Semaphore:
    return request.app.state.sandbox_semaphore


def get_ttl_cache(request: Request) -> SimpleTtlCache:
    return request.app.state.ttl_cache


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check(
    classifier: LexicalPhishingClassifier = Depends(get_classifier),
    visual_matcher: BrandVisualMatcher = Depends(get_visual_matcher),
) -> HealthResponse:
    """Return system operational and model readiness metrics."""
    tabular_loaded = classifier.model is not None
    brand_vault_loaded = len(visual_matcher.brand_meta) > 0

    status_str = "ok" if (tabular_loaded and brand_vault_loaded) else "degraded"

    device_str = "cpu"
    if visual_matcher.embedder and hasattr(visual_matcher.embedder, "device"):
        device_str = str(visual_matcher.embedder.device)

    return HealthResponse(
        status=status_str,
        tabular_model_loaded=tabular_loaded,
        brand_vault_loaded=brand_vault_loaded,
        indexed_brands_count=len(visual_matcher.brand_meta),
        device=device_str,
        version="1.0.0",
    )


@app.get("/api/v1/brands", tags=["Brands"])
async def list_monitored_brands(
    visual_matcher: BrandVisualMatcher = Depends(get_visual_matcher),
) -> list[dict]:
    """Return all protected brands and their canonical official domains."""
    return [b.model_dump() for b in visual_matcher.brand_meta.values()]


@app.get("/api/v1/history", response_model=list[ScanHistoryItem], tags=["History"])
async def get_scan_history(
    limit: int = Query(default=20, ge=1, le=100),
    history_store: ScanHistoryStore = Depends(get_history_store),
) -> list[ScanHistoryItem]:
    """Fetch the most recent URL scan results from the audit database."""
    return await history_store.get_recent(limit=limit)


@app.post("/api/v1/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_url(
    req: AnalyzeRequest,
    unwrapper: RedirectUnwrapper = Depends(get_unwrapper),
    sandbox: BrowserSandbox = Depends(get_sandbox),
    feature_extractor: TabularFeatureExtractor = Depends(get_feature_extractor),
    classifier: LexicalPhishingClassifier = Depends(get_classifier),
    visual_matcher: BrandVisualMatcher = Depends(get_visual_matcher),
    fusion: MultiModalRiskFusion = Depends(get_fusion),
    history_store: ScanHistoryStore = Depends(get_history_store),
    sandbox_semaphore: asyncio.Semaphore = Depends(get_sandbox_semaphore),
    cache: SimpleTtlCache = Depends(get_ttl_cache),
) -> AnalyzeResponse:
    """End-to-end multi-modal URL analysis for phishing and brand impersonation defense."""
    start_time = time.perf_counter()

    # 1. Validate scheme strictly
    if not (req.url.startswith("http://") or req.url.startswith("https://")):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="URL scheme must be http:// or https://",
        )

    # 2. Check in-memory TTL cache
    norm_url = RedirectUnwrapper.normalize_url(req.url)
    cache_key = (norm_url, req.skip_sandbox)

    if not req.force_rescan:
        cached_resp = await cache.get(cache_key)
        if cached_resp is not None:
            resp_copy = cached_resp.model_copy(update={"cached": True})
            if not req.include_screenshot and resp_copy.sandbox is not None:
                resp_copy.sandbox = resp_copy.sandbox.model_copy(update={"screenshot_bytes_b64": None})
            return resp_copy

    # 3. Step 1: Unravel redirect hops
    redirect_chain = await unwrapper.unwrap(req.url, max_hops=req.max_hops)

    # 4. Step 2: Headless Playwright sandbox execution
    sandbox_result = None
    if not req.skip_sandbox:
        async with sandbox_semaphore:
            sandbox_result = await sandbox.render(redirect_chain.final_url)

    # 5. Step 3: Run Tabular ML and Visual CV concurrently in non-blocking threads
    dom_signals = sandbox_result.dom_signals if sandbox_result else None
    screenshot_bytes = None
    if sandbox_result and sandbox_result.screenshot_bytes_b64:
        try:
            screenshot_bytes = base64.b64decode(sandbox_result.screenshot_bytes_b64)
        except Exception as e:
            logger.warning("Failed to decode screenshot base64: %s", e)

    async def run_tabular():
        feats, whois_failed, ssl_failed = await feature_extractor.extract(
            redirect_chain.final_url, redirect_chain, dom_signals
        )
        return await asyncio.to_thread(
            classifier.predict, feats, whois_failed, ssl_failed
        )

    async def run_visual():
        return await asyncio.to_thread(
            visual_matcher.match, redirect_chain.final_url, None, screenshot_bytes
        )

    lexical_result, visual_result = await asyncio.gather(run_tabular(), run_visual())

    # 6. Step 4: Multi-Modal Risk Fusion
    fusion_summary = fusion.fuse(
        lexical_result=lexical_result,
        visual_result=visual_result,
        redirect_chain=redirect_chain,
        dom_signals=dom_signals,
        skip_sandbox=req.skip_sandbox,
    )

    elapsed_ms = int(round((time.perf_counter() - start_time) * 1000))
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    req_id = str(uuid.uuid4())

    # 7. Audit log persistence
    history_item = ScanHistoryItem(
        request_id=req_id,
        analyzed_at=now_iso,
        original_url=req.url,
        final_url=redirect_chain.final_url,
        verdict=fusion_summary.verdict,
        risk_score_pct=fusion_summary.risk_score_pct,
        matched_brand_name=visual_result.matched_brand_name if visual_result else None,
        impersonation_detected=visual_result.impersonation_detected if visual_result else False,
        processing_time_ms=elapsed_ms,
    )
    await history_store.log_scan(history_item)

    # 8. Cache response (stores full data including screenshot for subsequent queries)
    response_for_cache = AnalyzeResponse(
        request_id=req_id,
        analyzed_at=now_iso,
        original_url=req.url,
        final_url=redirect_chain.final_url,
        cached=False,
        processing_time_ms=elapsed_ms,
        fusion=fusion_summary,
        redirect_chain=redirect_chain,
        sandbox=sandbox_result,
        lexical_ml=lexical_result,
        visual_cv=visual_result,
    )
    await cache.set(cache_key, response_for_cache)

    # 9. Format returned response according to include_screenshot flag
    if not req.include_screenshot and sandbox_result is not None:
        sandbox_result = sandbox_result.model_copy(update={"screenshot_bytes_b64": None})

    return AnalyzeResponse(
        request_id=req_id,
        analyzed_at=now_iso,
        original_url=req.url,
        final_url=redirect_chain.final_url,
        cached=False,
        processing_time_ms=elapsed_ms,
        fusion=fusion_summary,
        redirect_chain=redirect_chain,
        sandbox=sandbox_result,
        lexical_ml=lexical_result,
        visual_cv=visual_result,
    )
