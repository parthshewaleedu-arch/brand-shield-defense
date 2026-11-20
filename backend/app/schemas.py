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

