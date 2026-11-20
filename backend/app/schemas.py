from __future__ import annotations

import re
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    """Incoming request model for URL analysis."""
    url: str = Field(..., description="Target URL to inspect (must be valid http or https)")
    skip_sandbox: bool = Field(default=False, description="Whether to bypass headless browser sandbox rendering")
    max_hops: int = Field(default=10, ge=1, le=30, description="Maximum number of redirect hops to follow")
    force_rescan: bool = Field(default=False, description="Bypass cache and force re-scan")
    include_screenshot: bool = Field(
        default=False, description="Whether to include raw base64 screenshot in response"
    )

    @field_validator("url")
    @classmethod
    def validate_target_url(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("URL cannot be empty")
        # Ensure scheme is strictly http or https
        if not re.match(r"^https?://", cleaned, re.IGNORECASE):
            raise ValueError("URL scheme must be http:// or https://")
        return cleaned


class RedirectHop(BaseModel):
    """Represents a single step in a multi-hop redirect chain."""
    hop_number: int = Field(..., ge=1, description="Hop index starting from 1")
    url: str = Field(..., description="URL visited at this hop")
    status_code: Optional[int] = Field(default=None, description="HTTP response status code, if reached")
    mechanism: Literal["initial", "http_redirect", "meta_refresh", "js_redirect", "final"] = Field(
        ..., description="Redirect or landing mechanism at this hop"
    )
    response_headers: dict[str, str] = Field(
        default_factory=dict, description="Sanitized HTTP response headers with lowercase keys"
    )


class RedirectChainResult(BaseModel):
    """Result of tracing all redirect hops from original to final landing URL."""
    original_url: str = Field(..., description="Original input URL")
    final_url: str = Field(..., description="Final landing URL after tracing redirects")
    registered_domain: str = Field(..., description="Registered root domain (eTLD+1) of final URL")
    hops: list[RedirectHop] = Field(default_factory=list, description="Chronological list of hops")
    total_hops: int = Field(default=0, ge=0, description="Total number of hops traversed")
    has_shortener: bool = Field(default=False, description="Whether any hop domain is a known URL shortener")
    redirect_loop_detected: bool = Field(default=False, description="Whether a circular redirect loop was detected")
    errors: list[str] = Field(default_factory=list, description="Non-fatal warnings or error messages")


class DomSignals(BaseModel):
    """Structural, behavioral, and security signals extracted from rendered page DOM."""
    page_title: Optional[str] = Field(default=None, description="Rendered HTML document title")
    final_rendered_url: str = Field(..., description="Final browser location URL")
    has_password_input: bool = Field(default=False, description="Whether page contains input[type='password']")
    password_input_count: int = Field(default=0, ge=0, description="Count of password input fields")
    form_count: int = Field(default=0, ge=0, description="Total number of HTML form tags")
    external_form_action_count: int = Field(default=0, ge=0, description="Forms with actions pointing to external hosts")
    suspicious_form_actions: list[str] = Field(
        default_factory=list, description="List of suspicious external or malformed form action URLs"
    )
    iframe_count: int = Field(default=0, ge=0, description="Total count of iframe tags")
    hidden_iframe_count: int = Field(default=0, ge=0, description="Count of hidden or zero-dimension iframes")
    script_count: int = Field(default=0, ge=0, description="Total number of script tags")
    external_script_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Ratio of external scripts to total scripts")
    favicon_url: Optional[str] = Field(default=None, description="Resolved URL of page favicon if present")
    inline_event_handler_count: int = Field(
        default=0, ge=0, description="Count of inline event handlers (onclick, onload, etc.)"
    )
    obfuscated_js_heuristic: bool = Field(
        default=False, description="Whether obfuscation heuristics were triggered in script content"
    )


class SandboxResult(BaseModel):
    """Result of rendering a target URL inside the headless Playwright sandbox."""
    success: bool = Field(..., description="Whether headless browser rendering succeeded")
    final_url: str = Field(..., description="Final URL rendered by browser")
    http_status: Optional[int] = Field(default=None, description="Main document HTTP response status code")
    dom_signals: DomSignals = Field(..., description="Signals extracted from page DOM")
    screenshot_bytes_b64: Optional[str] = Field(default=None, description="Base64-encoded PNG screenshot (1280x720)")
    screenshot_sha256: Optional[str] = Field(default=None, description="SHA-256 hash of screenshot bytes")
    render_time_ms: int = Field(default=0, ge=0, description="Total render latency in milliseconds")
    error: Optional[str] = Field(default=None, description="Error message if rendering failed")


class ShapReason(BaseModel):
    """Plain-English explanation for a single SHAP feature contribution."""
    feature_name: str = Field(..., description="Canonical name of feature contributing to risk score")
    feature_value: float | int | str = Field(..., description="Observed feature value for this target")
    shap_impact: float = Field(..., description="Signed SHAP value indicating impact on phishing score")
    direction: Literal["increases_risk", "decreases_risk"] = Field(
        ..., description="Direction of contribution: increases_risk or decreases_risk"
    )
    explanation: str = Field(..., description="Concise, plain-English explanation for end users")


class LexicalMlResult(BaseModel):
    """Inference result from the Tabular ML & SHAP Explainability Engine (Module 3)."""
    phishing_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated phishing likelihood clamped between 0.0 and 1.0"
    )
    risk_label: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="Risk tier: LOW (< 0.35), MEDIUM (0.35-0.69), HIGH (>= 0.70)"
    )
    extracted_features: dict[str, float | int] = Field(
        ..., description="Complete dictionary of extracted canonical features"
    )
    top_reasons: list[ShapReason] = Field(
        default_factory=list, description="Top 3 SHAP feature explanations ranked by impact"
    )
    whois_lookup_failed: bool = Field(
        default=False, description="True if WHOIS domain query timed out or failed"
    )
    ssl_lookup_failed: bool = Field(
        default=False, description="True if SSL certificate inspection timed out or failed"
    )
    model_version: str = Field(..., description="Version identifier of the serialized model artifact")


class BrandRecord(BaseModel):
    """Metadata record representing a protected brand in the Brand Vault."""
    brand_id: str = Field(..., description="Unique slug for the brand (e.g., 'sbi', 'hdfc', 'google')")
    brand_name: str = Field(..., description="Human-readable brand name (e.g., 'State Bank of India')")
    official_domains: list[str] = Field(
        ..., min_length=1, description="List of canonical registered domains owned by this brand"
    )
    category: Literal["banking", "payments", "tech", "ecommerce", "social", "campus", "other"] = Field(
        ..., description="Sector/category of the brand"
    )
    reference_images: list[str] = Field(
        default_factory=list, description="Relative file paths of reference images inside data/brand_vault/images/"
    )


class VisualMatchCandidate(BaseModel):
    """Candidate match with similarity metrics against a reference brand."""
    brand_id: str = Field(..., description="Brand ID of the candidate match")
    brand_name: str = Field(..., description="Brand display name")
    cosine_similarity: float = Field(..., ge=0.0, le=1.0, description="MobileNetV3 deep embedding cosine similarity")
    phash_distance: int = Field(..., ge=0, le=64, description="64-bit perceptual hash Hamming distance")
    phash_similarity: float = Field(..., ge=0.0, le=1.0, description="Perceptual hash similarity (1.0 - dist / 64.0)")
    combined_visual_score: float = Field(
        ..., ge=0.0, le=1.0, description="Weighted fusion score (0.75 * cosine + 0.25 * phash_sim)"
    )


class VisualCvResult(BaseModel):
    """Inference result from the Visual Brand-Impersonation Engine (Module 1)."""
    matched_brand_id: Optional[str] = Field(default=None, description="Brand ID of highest matching candidate, if any")
    matched_brand_name: Optional[str] = Field(default=None, description="Display name of highest matching candidate")
    official_domains: list[str] = Field(
        default_factory=list, description="Official registered domains of the matched brand"
    )
    cosine_similarity: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Highest cosine similarity against reference embeddings"
    )
    phash_distance: Optional[int] = Field(
        default=None, ge=0, le=64, description="Perceptual hash Hamming distance for the top match"
    )
    combined_visual_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fused visual similarity score (0.0 to 1.0)"
    )
    domain_is_official: bool = Field(
        default=False, description="Whether target domain is confirmed in official domains of matched brand"
    )
    impersonation_detected: bool = Field(
        default=False, description="Whether visual brand impersonation was confirmed on an unauthorized domain"
    )
    top_candidates: list[VisualMatchCandidate] = Field(
        default_factory=list, description="Top 3 brand matches ranked descending by combined visual score"
    )
    inference_time_ms: int = Field(default=0, ge=0, description="Total visual inference latency in milliseconds")
    skipped_reason: Optional[str] = Field(
        default=None, description="Reason if visual analysis was skipped (e.g. blank screenshot, no screenshot)"
    )


class FusionSummary(BaseModel):
    """Unified risk evaluation synthesized across Tabular ML, Visual CV, and Redirect/DOM analysis."""
    verdict: Literal["SAFE", "SUSPICIOUS", "CRITICAL_PHISHING"] = Field(
        ..., description="Final security classification: SAFE (<0.35), SUSPICIOUS (0.35-0.69), CRITICAL_PHISHING (>=0.70)"
    )
    final_risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Fused aggregate risk probability clamped between 0.0 and 1.0"
    )
    risk_score_pct: int = Field(
        ..., ge=0, le=100, description="Integer percentage representation of the risk score (0 to 100)"
    )
    recommended_action: Literal["ALLOW", "WARN", "BLOCK_CREDENTIALS"] = Field(
        ..., description="Actionable guidance for the browser extension and security gateway"
    )
    override_triggered: bool = Field(
        default=False, description="Whether a hard override (e.g. Visual Brand Impersonation) dictated the verdict"
    )
    override_reason: Optional[str] = Field(
        default=None, description="Detailed explanation if an override rule was triggered"
    )
    component_scores: dict[str, float] = Field(
        default_factory=dict, description="Individual component scores (tabular_ml, visual_cv, redirect_dom)"
    )
    headline: str = Field(
        ..., description="One-sentence plain-English summary suitable for user-facing security overlays"
    )


class AnalyzeResponse(BaseModel):
    """Complete diagnostic payload returned by POST /api/v1/analyze."""
    request_id: str = Field(..., description="Unique UUID identifier for this scan request")
    analyzed_at: str = Field(..., description="ISO-8601 UTC timestamp of analysis completion")
    original_url: str = Field(..., description="Target URL originally submitted for analysis")
    final_url: str = Field(..., description="Final URL resolved after following all redirect hops")
    cached: bool = Field(default=False, description="True if response was served from in-memory TTL cache")
    processing_time_ms: int = Field(default=0, ge=0, description="Total end-to-end analysis latency in milliseconds")
    fusion: FusionSummary = Field(..., description="Aggregated multi-modal fusion verdict and risk scores")
    redirect_chain: RedirectChainResult = Field(..., description="Complete trace of all redirect hops")
    sandbox: Optional[SandboxResult] = Field(default=None, description="Headless browser execution results if executed")
    lexical_ml: LexicalMlResult = Field(..., description="Tabular ML and SHAP explainability inference results")
    visual_cv: Optional[VisualCvResult] = Field(default=None, description="Computer Vision brand match inference results")


class ScanHistoryItem(BaseModel):
    """Audit log item representing an analyzed URL stored in SQLite."""
    request_id: str = Field(..., description="UUID identifier of the scan")
    analyzed_at: str = Field(..., description="ISO-8601 UTC timestamp")
    original_url: str = Field(..., description="Original requested URL")
    final_url: str = Field(..., description="Final landing URL")
    verdict: Literal["SAFE", "SUSPICIOUS", "CRITICAL_PHISHING"] = Field(..., description="Final classification verdict")
    risk_score_pct: int = Field(..., ge=0, le=100, description="Risk score percentage")
    matched_brand_name: Optional[str] = Field(default=None, description="Matched brand name if impersonation detected")
    impersonation_detected: bool = Field(default=False, description="Whether visual impersonation was confirmed")
    processing_time_ms: int = Field(default=0, ge=0, description="Processing latency in ms")


class HealthResponse(BaseModel):
    """System diagnostic response returned by GET /api/v1/health."""
    status: Literal["ok", "degraded"] = Field(..., description="Operational status of backend service")
    tabular_model_loaded: bool = Field(..., description="Whether LightGBM tabular classifier is loaded in memory")
    brand_vault_loaded: bool = Field(..., description="Whether Brand Vault index and metadata are loaded in memory")
    indexed_brands_count: int = Field(default=0, ge=0, description="Number of brands indexed in Brand Vault")
    device: str = Field(..., description="Computation device utilized for PyTorch inference (cpu or cuda)")
    version: str = Field(default="1.0.0", description="API version identifier")



