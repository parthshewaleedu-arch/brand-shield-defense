from __future__ import annotations

import asyncio
import datetime
import ipaddress
import logging
import math
import os
import re
import socket
import ssl
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tldextract

from backend.app.schemas import (
    DomSignals,
    LexicalMlResult,
    RedirectChainResult,
    ShapReason,
)

logger = logging.getLogger("brandshield.lexical_ml")

# Canonical 27 feature columns shared strictly between training and inference
FEATURE_COLUMNS: list[str] = [
    # 1. URL Lexical Topology (15 features)
    "url_length",
    "hostname_length",
    "path_length",
    "subdomain_depth",
    "dot_count",
    "hyphen_count",
    "at_symbol_present",
    "ip_in_hostname",
    "digit_ratio",
    "shannon_entropy",
    "suspicious_tld_flag",
    "brand_keyword_in_subdomain",
    "punycode_flag",
    "is_https",
    "query_param_count",
    # 2. Domain & SSL Telemetry (4 features)
    "domain_age_days",
    "ssl_valid",
    "ssl_cert_age_days",
    "ssl_is_self_signed",
    # 3. Phase 1 Redirect & DOM Signals (8 features)
    "total_hops",
    "has_shortener",
    "redirect_loop_detected",
    "has_password_input",
    "external_form_action_count",
    "hidden_iframe_count",
    "external_script_ratio",
    "obfuscated_js_heuristic",
]

# High-risk TLDs heavily abused by zero-day phishing infrastructure
SUSPICIOUS_TLDS: set[str] = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "buzz", "icu", "work",
    "zip", "click", "rest", "cam", "fit", "surf", "casa", "fun", "monster",
    "link", "gdn", "bid", "loan", "men", "date", "racing", "win"
}

# High-value brand names & credential harvesting tokens
BRAND_KEYWORDS: set[str] = {
    "sbi", "onlinesbi", "hdfc", "hdfcbank", "icici", "icicibank", "axis", "axisbank",
    "paytm", "phonepe", "gpay", "upi", "google", "microsoft", "amazon", "apple",
    "netflix", "github", "linkedin", "instagram", "facebook", "twitter", "paypal",
    "login", "signin", "verify", "secure", "update", "banking", "account", "support",
    "portal", "auth", "validation", "wallet", "ebanking", "kyc"
}

# Template mapper generating human-readable security warnings for SHAP contributors
EXPLANATION_TEMPLATES: dict[str, dict[str, Callable[[float | int | str], str]]] = {
    "domain_age_days": {
        "increases_risk": lambda v: f"Domain was registered very recently ({int(v)} days ago)" if float(v) >= 0 else "Domain age could not be verified via WHOIS (new or hidden registry)",
        "decreases_risk": lambda v: f"Domain has an established registration history ({int(v)} days old)",
    },
    "has_password_input": {
        "increases_risk": lambda v: "Page renders a sensitive password credential input form",
        "decreases_risk": lambda v: "No password input fields detected on page",
    },
    "external_form_action_count": {
        "increases_risk": lambda v: f"Form action targets an external third-party domain ({int(v)} external form{'s' if float(v) > 1 else ''})",
        "decreases_risk": lambda v: "Form submissions are routed internally to the verified domain",
    },
    "suspicious_tld_flag": {
        "increases_risk": lambda v: "URL uses a high-risk generic top-level domain frequently abused by phishing campaigns",
        "decreases_risk": lambda v: "URL uses a standard reputable top-level domain",
    },
    "brand_keyword_in_subdomain": {
        "increases_risk": lambda v: "Subdomain embeds sensitive brand or authentication tokens to simulate legitimacy",
        "decreases_risk": lambda v: "Subdomain does not contain brand impersonation keywords",
    },
    "shannon_entropy": {
        "increases_risk": lambda v: f"URL exhibits unusually high character randomness (entropy {float(v):.2f})",
        "decreases_risk": lambda v: "URL structure exhibits normal lexical predictability",
    },
    "ip_in_hostname": {
        "increases_risk": lambda v: "Target uses a raw IP address instead of an authenticated domain name",
        "decreases_risk": lambda v: "Target uses a standard domain hostname",
    },
    "punycode_flag": {
        "increases_risk": lambda v: "Domain utilizes punycode encoding (potential IDN homograph attack)",
        "decreases_risk": lambda v: "Standard character set without homograph punycode",
    },
    "total_hops": {
        "increases_risk": lambda v: f"URL undergoes multiple redirect hops ({int(v)} hops) obscuring final destination",
        "decreases_risk": lambda v: "Direct URL navigation with minimal redirects",
    },
    "has_shortener": {
        "increases_risk": lambda v: "URL employs a link shortener service to disguise the landing page",
        "decreases_risk": lambda v: "URL destination is directly visible without shortener obfuscation",
    },
    "redirect_loop_detected": {
        "increases_risk": lambda v: "A circular redirection loop was encountered",
        "decreases_risk": lambda v: "Linear redirection chain without loops",
    },
    "hidden_iframe_count": {
        "increases_risk": lambda v: f"Page embeds hidden or zero-dimension iframes ({int(v)} detected)",
        "decreases_risk": lambda v: "No hidden iframe tags detected in DOM",
    },
    "obfuscated_js_heuristic": {
        "increases_risk": lambda v: "Client scripts contain obfuscation or packing heuristics (eval/unescape)",
        "decreases_risk": lambda v: "Client scripts follow clean, unobfuscated code patterns",
    },
    "external_script_ratio": {
        "increases_risk": lambda v: f"High proportion of external third-party script resources ({float(v)*100:.0f}%)",
        "decreases_risk": lambda v: "Low reliance on external script origins",
    },
    "is_https": {
        "increases_risk": lambda v: "Connection is unencrypted HTTP without SSL/TLS protection",
        "decreases_risk": lambda v: "Connection is encrypted via HTTPS",
    },
    "ssl_valid": {
        "increases_risk": lambda v: "SSL certificate is invalid, untrusted, or expired",
        "decreases_risk": lambda v: "SSL certificate is valid and issued by a trusted CA",
    },
    "ssl_is_self_signed": {
        "increases_risk": lambda v: "SSL certificate is self-signed rather than validated by an accredited authority",
        "decreases_risk": lambda v: "SSL certificate is signed by an accredited CA",
    },
    "ssl_cert_age_days": {
        "increases_risk": lambda v: f"SSL certificate is newly issued ({int(v)} days old)" if float(v) >= 0 else "SSL certificate age unavailable",
        "decreases_risk": lambda v: f"SSL certificate is well-established ({int(v)} days old)",
    },
    "digit_ratio": {
        "increases_risk": lambda v: f"Unusually high ratio of digits in URL ({float(v)*100:.1f}%)",
        "decreases_risk": lambda v: "Normal alphanumeric character balance",
    },
    "url_length": {
        "increases_risk": lambda v: f"Excessively long URL string ({int(v)} characters)",
        "decreases_risk": lambda v: "Compact URL length",
    },
    "hostname_length": {
        "increases_risk": lambda v: f"Unusually long hostname ({int(v)} characters)",
        "decreases_risk": lambda v: "Standard hostname length",
    },
    "path_length": {
        "increases_risk": lambda v: f"Deep or complex URL path structure ({int(v)} characters)",
        "decreases_risk": lambda v: "Standard path structure",
    },
    "subdomain_depth": {
        "increases_risk": lambda v: f"High subdomain nesting depth ({int(v)} levels)",
        "decreases_risk": lambda v: "Standard subdomain hierarchy",
    },
    "dot_count": {
        "increases_risk": lambda v: f"High count of '.' separators in URL ({int(v)} dots)",
        "decreases_risk": lambda v: "Normal dot separator count",
    },
    "hyphen_count": {
        "increases_risk": lambda v: f"Frequent hyphenation in URL ({int(v)} hyphens)",
        "decreases_risk": lambda v: "Low hyphen count",
    },
    "at_symbol_present": {
        "increases_risk": lambda v: "URL contains '@' symbol typically used to disguise destination credentials",
        "decreases_risk": lambda v: "No '@' character in URL",
    },
    "query_param_count": {
        "increases_risk": lambda v: f"Complex query parameter payload ({int(v)} parameters)",
        "decreases_risk": lambda v: "Standard query parameters",
    },
}


def calculate_shannon_entropy(text: str) -> float:
    """Calculate the Shannon entropy of a string."""
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return round(entropy, 4)


def is_ip_address(hostname: str) -> bool:
    """Check if the provided hostname is a valid IPv4 or IPv6 address."""
    clean_host = hostname.split(":")[0].strip("[]")
    try:
        ipaddress.ip_address(clean_host)
        return True
    except ValueError:
        return False


class TabularFeatureExtractor:
    """
    Asynchronous, deterministic feature extractor constructing the canonical
    27-element feature vector from URL lexical structure, WHOIS, SSL, and Phase 1 telemetry.
    """

    def __init__(self, lookup_timeout_seconds: float = 3.0) -> None:
        self.lookup_timeout_seconds = lookup_timeout_seconds

    @staticmethod
    def extract_url_lexical(url: str) -> dict[str, float | int]:
        """Extract the 15 URL lexical features from target URL."""
        parsed = urllib.parse.urlparse(url)
        hostname = (parsed.hostname or parsed.netloc or "").split(":")[0].lower()
        path = parsed.path or ""
        scheme = parsed.scheme.lower()

        ext = tldextract.extract(url)
        subdomain = (ext.subdomain or "").lower()
        tld = (ext.suffix or "").lower()

        # 1. url_length
        url_length = len(url)

        # 2. hostname_length
        hostname_length = len(hostname)

        # 3. path_length
        path_length = len(path)

        # 4. subdomain_depth
        subdomain_parts = [p for p in subdomain.split(".") if p]
        subdomain_depth = len(subdomain_parts)

        # 5. dot_count
        dot_count = url.count(".")

        # 6. hyphen_count
        hyphen_count = url.count("-")

        # 7. at_symbol_present
        at_symbol_present = 1 if "@" in url else 0

        # 8. ip_in_hostname
        ip_in_hostname = 1 if is_ip_address(hostname) else 0

        # 9. digit_ratio
        digit_count = sum(1 for c in url if c.isdigit())
        digit_ratio = round(digit_count / url_length, 4) if url_length > 0 else 0.0

        # 10. shannon_entropy
        shannon_entropy = calculate_shannon_entropy(url)

        # 11. suspicious_tld_flag
        last_tld_token = tld.split(".")[-1] if tld else ""
        suspicious_tld_flag = 1 if last_tld_token in SUSPICIOUS_TLDS else 0

        # 12. brand_keyword_in_subdomain
        brand_keyword_in_subdomain = 0
        if subdomain:
            subdomain_tokens = set(re.split(r"[-_.]", subdomain))
            if any(k in subdomain_tokens or k in subdomain for k in BRAND_KEYWORDS):
                brand_keyword_in_subdomain = 1

        # 13. punycode_flag
        punycode_flag = 1 if "xn--" in hostname else 0

        # 14. is_https
        is_https = 1 if scheme == "https" else 0

        # 15. query_param_count
        query_params = urllib.parse.parse_qsl(parsed.query)
        query_param_count = len(query_params)

        return {
            "url_length": url_length,
            "hostname_length": hostname_length,
            "path_length": path_length,
            "subdomain_depth": subdomain_depth,
            "dot_count": dot_count,
            "hyphen_count": hyphen_count,
            "at_symbol_present": at_symbol_present,
            "ip_in_hostname": ip_in_hostname,
            "digit_ratio": digit_ratio,
            "shannon_entropy": shannon_entropy,
            "suspicious_tld_flag": suspicious_tld_flag,
            "brand_keyword_in_subdomain": brand_keyword_in_subdomain,
            "punycode_flag": punycode_flag,
            "is_https": is_https,
            "query_param_count": query_param_count,
        }

    @staticmethod
    def _sync_whois_lookup(domain: str) -> tuple[int, bool]:
        """Perform blocking WHOIS lookup; returns (domain_age_days, lookup_failed)."""
        if not domain or is_ip_address(domain):
            return -1, True

        try:
            import whois

            w = whois.whois(domain)
            creation_date = w.creation_date
            if isinstance(creation_date, (list, tuple)):
                creation_date = creation_date[0] if creation_date else None

            if isinstance(creation_date, datetime.datetime):
                if creation_date.tzinfo is not None:
                    creation_date = creation_date.replace(tzinfo=None)
                now = datetime.datetime.now()
                age = (now - creation_date).days
                return max(0, age), False

            return -1, False
        except Exception as exc:
            logger.debug("WHOIS query failed for %s: %s", domain, exc)
            return -1, True

    @staticmethod
    def _sync_ssl_lookup(hostname: str, port: int = 443) -> tuple[int, int, int, bool]:
        """Perform blocking SSL inspection; returns (ssl_valid, ssl_cert_age_days, ssl_is_self_signed, failed)."""
        if not hostname or is_ip_address(hostname):
            return 0, -1, 0, True

        try:
            context = ssl.create_default_context()
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED

            with socket.create_connection((hostname, port), timeout=2.5) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    if not cert:
                        return 0, -1, 0, False

                    not_before = cert.get("notBefore")
                    cert_age_days = -1
                    if not_before:
                        try:
                            # Format e.g. "May  5 00:00:00 2026 GMT"
                            issue_dt = datetime.datetime.strptime(not_before, "%b %d %H:%M:%S %Y %Z")
                            now_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                            cert_age_days = max(0, (now_utc - issue_dt).days)
                        except Exception:
                            pass

                    issuer = dict(x[0] for x in cert.get("issuer", []))
                    subject = dict(x[0] for x in cert.get("subject", []))
                    is_self_signed = 1 if issuer and issuer == subject else 0

                    return 1, cert_age_days, is_self_signed, False

        except ssl.SSLCertVerificationError as exc:
            # Expired, untrusted, or self-signed cert
            is_self_signed = 1 if "self-signed" in str(exc).lower() else 0
            return 0, -1, is_self_signed, False
        except Exception as exc:
            logger.debug("SSL query failed for %s: %s", hostname, exc)
            return 0, -1, 0, True

    async def extract_domain_ssl(self, url: str) -> tuple[dict[str, float | int], bool, bool]:
        """
        Extract domain age and SSL metrics asynchronously with strict timeouts.
        Returns (features_dict, whois_failed, ssl_failed).
        """
        parsed = urllib.parse.urlparse(url)
        hostname = (parsed.hostname or parsed.netloc or "").split(":")[0]

        ext = tldextract.extract(url)
        if hasattr(ext, "top_domain_under_public_suffix"):
            domain = ext.top_domain_under_public_suffix
        else:
            domain = ext.registered_domain
        if not domain:
            domain = hostname

        # 1. Asynchronous WHOIS lookup
        whois_task = asyncio.to_thread(self._sync_whois_lookup, domain)
        try:
            domain_age_days, whois_failed = await asyncio.wait_for(
                whois_task, timeout=self.lookup_timeout_seconds
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.debug("Asynchronous WHOIS lookup timed out or failed for %s: %s", domain, exc)
            domain_age_days = -1
            whois_failed = True

        # 2. Asynchronous SSL inspection
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if parsed.scheme == "https":
            ssl_task = asyncio.to_thread(self._sync_ssl_lookup, hostname, port)
            try:
                ssl_valid, ssl_cert_age_days, ssl_is_self_signed, ssl_failed = await asyncio.wait_for(
                    ssl_task, timeout=self.lookup_timeout_seconds
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.debug("Asynchronous SSL inspection timed out or failed for %s: %s", hostname, exc)
                ssl_valid = 0
                ssl_cert_age_days = -1
                ssl_is_self_signed = 0
                ssl_failed = True
        else:
            ssl_valid = 0
            ssl_cert_age_days = -1
            ssl_is_self_signed = 0
            ssl_failed = False

        features = {
            "domain_age_days": domain_age_days,
            "ssl_valid": ssl_valid,
            "ssl_cert_age_days": ssl_cert_age_days,
            "ssl_is_self_signed": ssl_is_self_signed,
        }
        return features, whois_failed, ssl_failed

    @staticmethod
    def extract_from_signals(
        redirect: Optional[RedirectChainResult] = None,
        dom: Optional[DomSignals] = None,
    ) -> dict[str, float | int]:
        """Extract the 8 features derived from Phase 1 redirect and DOM telemetry."""
        total_hops = redirect.total_hops if redirect else 1
        has_shortener = 1 if (redirect and redirect.has_shortener) else 0
        redirect_loop_detected = 1 if (redirect and redirect.redirect_loop_detected) else 0

        has_password_input = 1 if (dom and dom.has_password_input) else 0
        external_form_action_count = dom.external_form_action_count if dom else 0
        hidden_iframe_count = dom.hidden_iframe_count if dom else 0
        external_script_ratio = dom.external_script_ratio if dom else 0.0
        obfuscated_js_heuristic = 1 if (dom and dom.obfuscated_js_heuristic) else 0

        return {
            "total_hops": total_hops,
            "has_shortener": has_shortener,
            "redirect_loop_detected": redirect_loop_detected,
            "has_password_input": has_password_input,
            "external_form_action_count": external_form_action_count,
            "hidden_iframe_count": hidden_iframe_count,
            "external_script_ratio": external_script_ratio,
            "obfuscated_js_heuristic": obfuscated_js_heuristic,
        }

    async def extract(
        self,
        url: str,
        redirect: Optional[RedirectChainResult] = None,
        dom: Optional[DomSignals] = None,
    ) -> tuple[dict[str, float | int], bool, bool]:
        """
        Extract the complete ordered 27-element canonical feature dictionary.
        Returns (features_dict, whois_lookup_failed, ssl_lookup_failed).
        """
        # Resolve final target URL if redirect chain is provided
        target_url = redirect.final_url if redirect and redirect.final_url else url

        # 1. Lexical features
        lexical_feats = self.extract_url_lexical(target_url)

        # 2. Domain & SSL telemetry
        domain_ssl_feats, whois_failed, ssl_failed = await self.extract_domain_ssl(target_url)

        # 3. Phase 1 Redirect & DOM telemetry
        signals_feats = self.extract_from_signals(redirect, dom)

        merged = {**lexical_feats, **domain_ssl_feats, **signals_feats}

        # Ensure all columns exist in the strict canonical order
        ordered_feats = {col: merged.get(col, 0) for col in FEATURE_COLUMNS}
        return ordered_feats, whois_failed, ssl_failed


class LexicalPhishingClassifier:
    """
    Production tabular inference engine for LightGBM/XGBoost with SHAP explainability.
    Loads serialized model artifacts and computes calibrated probabilities and top-3 SHAP explanations.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
        self.model = None
        self.explainer = None
        self.model_version: str = "tabular_v1.0.0"
        self.feature_columns: list[str] = FEATURE_COLUMNS

        if model_path:
            self.load(model_path)

    def load(self, model_path: Union[str, Path]) -> None:
        """Load trained model artifact and precompute shap.TreeExplainer."""
        import joblib
        import shap

        resolved_path = Path(model_path).resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {resolved_path}")

        artifact = joblib.load(resolved_path)

        if isinstance(artifact, dict) and "model" in artifact:
            self.model = artifact["model"]
            self.feature_columns = artifact.get("feature_columns", FEATURE_COLUMNS)
            self.model_version = artifact.get("model_version", "tabular_v1.0.0")
        else:
            self.model = artifact
            self.feature_columns = FEATURE_COLUMNS
            self.model_version = "tabular_v1.0.0"

        # Initialize TreeExplainer once at load time
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("Loaded model artifact %s (%s)", self.model_version, type(self.model).__name__)

    def _extract_shap_array(self, shap_raw: Any) -> np.ndarray:
        """
        Normalize SHAP output shape variations across versions:
        Handles list of arrays [class0, class1], 3D (1, n, 2), 2D (1, n), or 1D (n).
        Always returns a 1D float array of length len(self.feature_columns).
        """
        if isinstance(shap_raw, list):
            # Binary classification: [class_0_array, class_1_array]
            pos_class = shap_raw[1] if len(shap_raw) > 1 else shap_raw[0]
            arr = np.asarray(pos_class)
            if arr.ndim > 1:
                return arr[0].astype(float)
            return arr.astype(float)

        arr = np.asarray(shap_raw)
        if arr.ndim == 3:
            # Shape (1, n_features, 2) -> take positive class index 1
            return arr[0, :, 1].astype(float)
        elif arr.ndim == 2:
            # Shape (1, n_features)
            return arr[0].astype(float)
        elif arr.ndim == 1:
            return arr.astype(float)

        # Fallback to zeros if unexpected shape
        return np.zeros(len(self.feature_columns), dtype=float)

    def predict(
        self,
        features: dict[str, float | int],
        whois_lookup_failed: bool = False,
        ssl_lookup_failed: bool = False,
    ) -> LexicalMlResult:
        """Run tabular inference and compute top-3 plain-English SHAP risk explanations."""
        if self.model is None or self.explainer is None:
            raise RuntimeError("Classifier model is not loaded. Call load() before predict().")

        # Prepare 1xN feature vector in strict column order
        vector = np.array([[features.get(col, 0) for col in self.feature_columns]], dtype=float)

        # 1. Compute prediction probability
        proba = self.model.predict_proba(vector)
        phishing_probability = float(proba[0][1])
        clamped_prob = max(0.0, min(1.0, round(phishing_probability, 4)))

        # 2. Risk label categorization
        if clamped_prob < 0.35:
            risk_label = "LOW"
        elif clamped_prob < 0.70:
            risk_label = "MEDIUM"
        else:
            risk_label = "HIGH"

        # 3. Compute SHAP contributions
        raw_shap = self.explainer.shap_values(vector)
        shap_values = self._extract_shap_array(raw_shap)

        # 4. Rank features: prioritize positive impacts when suspicious/phishing, else largest absolute impacts
        indexed_impacts = list(enumerate(shap_values))
        if clamped_prob >= 0.35:
            # Positive SHAP values increase risk
            positives = [item for item in indexed_impacts if item[1] > 0]
            positives.sort(key=lambda x: x[1], reverse=True)
            if len(positives) >= 3:
                top_indices = positives[:3]
            else:
                # Backfill with largest absolute impacts
                remaining = [item for item in indexed_impacts if item not in positives]
                remaining.sort(key=lambda x: abs(x[1]), reverse=True)
                top_indices = (positives + remaining)[:3]
        else:
            # Low risk: rank by largest absolute impacts
            indexed_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            top_indices = indexed_impacts[:3]

        top_reasons: list[ShapReason] = []
        for idx, impact in top_indices:
            feat_name = self.feature_columns[idx]
            feat_val = features.get(feat_name, 0)
            direction = "increases_risk" if impact > 0 else "decreases_risk"

            template_entry = EXPLANATION_TEMPLATES.get(feat_name, {})
            expl_fn = template_entry.get(direction)
            if expl_fn:
                explanation = expl_fn(feat_val)
            else:
                formatted_feat = feat_name.replace("_", " ")
                explanation = f"Feature '{formatted_feat}' ({feat_val}) {direction.replace('_', ' ')}"

            top_reasons.append(
                ShapReason(
                    feature_name=feat_name,
                    feature_value=feat_val,
                    shap_impact=round(float(impact), 4),
                    direction=direction,
                    explanation=explanation,
                )
            )

        return LexicalMlResult(
            phishing_probability=clamped_prob,
            risk_label=risk_label,
            extracted_features=features,
            top_reasons=top_reasons,
            whois_lookup_failed=whois_lookup_failed,
            ssl_lookup_failed=ssl_lookup_failed,
            model_version=self.model_version,
        )
