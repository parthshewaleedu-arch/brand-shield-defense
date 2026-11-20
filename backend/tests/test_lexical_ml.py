from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from backend.app.core.lexical_ml import (
    FEATURE_COLUMNS,
    LexicalPhishingClassifier,
    TabularFeatureExtractor,
)
from backend.app.schemas import DomSignals, RedirectChainResult, RedirectHop
from backend.train_tabular import generate_bootstrap_dataset, train_and_compare


def test_lexical_feature_extraction_normal_url() -> None:
    extractor = TabularFeatureExtractor()
    url = "https://www.google.com/search?q=cybersecurity&hl=en"
    feats = extractor.extract_url_lexical(url)

    assert feats["url_length"] == len(url)
    assert feats["is_https"] == 1
    assert feats["ip_in_hostname"] == 0
    assert feats["punycode_flag"] == 0
    assert feats["suspicious_tld_flag"] == 0
    assert feats["brand_keyword_in_subdomain"] == 0
    assert feats["query_param_count"] == 2
    assert feats["shannon_entropy"] > 2.0
    assert feats["dot_count"] >= 2
    assert feats["hyphen_count"] == 0


def test_lexical_feature_extraction_raw_ip() -> None:
    extractor = TabularFeatureExtractor()
    url = "http://192.168.1.100:8080/secure/login"
    feats = extractor.extract_url_lexical(url)

    assert feats["is_https"] == 0
    assert feats["ip_in_hostname"] == 1
    assert feats["punycode_flag"] == 0
    assert feats["dot_count"] == 3


def test_lexical_feature_extraction_punycode() -> None:
    extractor = TabularFeatureExtractor()
    url = "https://xn--sbi-bank-52a.com/portal"
    feats = extractor.extract_url_lexical(url)

    assert feats["punycode_flag"] == 1
    assert feats["is_https"] == 1


def test_lexical_feature_extraction_suspicious_tld() -> None:
    extractor = TabularFeatureExtractor()
    url = "http://account-security-update.xyz/verify"
    feats = extractor.extract_url_lexical(url)

    assert feats["suspicious_tld_flag"] == 1
    assert feats["is_https"] == 0


def test_lexical_feature_extraction_brand_in_subdomain() -> None:
    extractor = TabularFeatureExtractor()
    url = "https://login.hdfc-verify.secure-banking.evil.xyz/auth"
    feats = extractor.extract_url_lexical(url)

    assert feats["brand_keyword_in_subdomain"] == 1
    assert feats["subdomain_depth"] >= 2
    assert feats["suspicious_tld_flag"] == 1


def test_lexical_feature_high_entropy() -> None:
    extractor = TabularFeatureExtractor()
    url = "https://target-portal.com/d9F#k8%L1!mZ7@pQ3"
    feats = extractor.extract_url_lexical(url)

    assert feats["shannon_entropy"] > 3.5


@pytest.mark.asyncio
async def test_whois_lookup_success_single_and_list_datetime() -> None:
    extractor = TabularFeatureExtractor()

    # 1. Single datetime object
    mock_whois_single = MagicMock()
    mock_whois_single.creation_date = datetime.datetime.now() - datetime.timedelta(days=1200)

    with patch("whois.whois", return_value=mock_whois_single):
        feats, whois_failed, _ = await extractor.extract_domain_ssl("https://example.com")
        assert whois_failed is False
        assert 1195 <= feats["domain_age_days"] <= 1205

    # 2. List of datetimes (common in whois responses)
    mock_whois_list = MagicMock()
    dt1 = datetime.datetime.now() - datetime.timedelta(days=500)
    dt2 = datetime.datetime.now() - datetime.timedelta(days=505)
    mock_whois_list.creation_date = [dt1, dt2]

    with patch("whois.whois", return_value=mock_whois_list):
        feats, whois_failed, _ = await extractor.extract_domain_ssl("https://example.com")
        assert whois_failed is False
        assert 495 <= feats["domain_age_days"] <= 505


@pytest.mark.asyncio
async def test_whois_lookup_failure_and_timeout() -> None:
    extractor = TabularFeatureExtractor(lookup_timeout_seconds=0.5)

    # 1. Exception raised
    with patch("whois.whois", side_effect=Exception("Connection refused")):
        feats, whois_failed, _ = await extractor.extract_domain_ssl("https://nonexistent-phish-domain.xyz")
        assert whois_failed is True
        assert feats["domain_age_days"] == -1

    # 2. Raw IP target (WHOIS bypassed cleanly)
    feats, whois_failed, _ = await extractor.extract_domain_ssl("http://192.168.1.50/login")
    assert whois_failed is True
    assert feats["domain_age_days"] == -1


@pytest.mark.asyncio
async def test_ssl_lookup_mocked() -> None:
    extractor = TabularFeatureExtractor()

    # Mock SSL certificate
    mock_cert = {
        "notBefore": "Jan  1 00:00:00 2026 GMT",
        "issuer": ((("organizationName", "DigiCert Inc"),),),
        "subject": ((("commonName", "example.com"),),),
    }

    mock_sock = MagicMock()
    mock_ssock = MagicMock()
    mock_ssock.getpeercert.return_value = mock_cert
    mock_ssock.__enter__.return_value = mock_ssock
    mock_sock.__enter__.return_value = mock_sock

    with patch("socket.create_connection", return_value=mock_sock), \
         patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.return_value = mock_ssock
        feats, _, ssl_failed = await extractor.extract_domain_ssl("https://example.com")

        assert ssl_failed is False
        assert feats["ssl_valid"] == 1
        assert feats["ssl_is_self_signed"] == 0
        assert feats["ssl_cert_age_days"] >= 0


def test_signals_integration() -> None:
    extractor = TabularFeatureExtractor()

    redirect_res = RedirectChainResult(
        original_url="http://short.ly/xyz",
        final_url="https://evil.xyz/login",
        registered_domain="evil.xyz",
        hops=[
            RedirectHop(hop_number=1, url="http://short.ly/xyz", status_code=301, mechanism="http_redirect"),
            RedirectHop(hop_number=2, url="https://evil.xyz/login", status_code=200, mechanism="final"),
        ],
        total_hops=2,
        has_shortener=True,
        redirect_loop_detected=False,
        errors=[],
    )

    dom_res = DomSignals(
        page_title="Fake Portal",
        final_rendered_url="https://evil.xyz/login",
        has_password_input=True,
        password_input_count=1,
        form_count=1,
        external_form_action_count=1,
        suspicious_form_actions=["https://credential-stealer.com/post"],
        iframe_count=1,
        hidden_iframe_count=1,
        script_count=4,
        external_script_ratio=0.75,
        favicon_url=None,
        inline_event_handler_count=2,
        obfuscated_js_heuristic=True,
    )

    signals_feats = extractor.extract_from_signals(redirect_res, dom_res)

    assert signals_feats["total_hops"] == 2
    assert signals_feats["has_shortener"] == 1
    assert signals_feats["redirect_loop_detected"] == 0
    assert signals_feats["has_password_input"] == 1
    assert signals_feats["external_form_action_count"] == 1
    assert signals_feats["hidden_iframe_count"] == 1
    assert signals_feats["external_script_ratio"] == 0.75
    assert signals_feats["obfuscated_js_heuristic"] == 1


def test_end_to_end_training_and_shap_inference(tmp_path: Path) -> None:
    import joblib

    # 1. Generate small bootstrap dataset for fast test execution
    X, y, feature_names = generate_bootstrap_dataset(n_samples=500, random_state=42)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 2. Train model and verify benchmarks
    champion_model, champion_name, metrics, _ = train_and_compare(X_train, X_test, y_train, y_test)
    assert metrics["accuracy"] >= 0.85
    assert metrics["precision"] >= 0.85
    assert metrics["pr_auc"] >= 0.85
    assert metrics["fnr"] <= 0.15

    # 3. Serialize artifact
    artifact_path = tmp_path / "test_tabular_model.joblib"
    artifact = {
        "model": champion_model,
        "feature_columns": FEATURE_COLUMNS,
        "model_version": "test_v1.0.0",
        "model_name": champion_name,
        "metrics": metrics,
    }
    joblib.dump(artifact, artifact_path)

    # 4. Load into LexicalPhishingClassifier
    classifier = LexicalPhishingClassifier(artifact_path)
    assert classifier.model is not None
    assert classifier.explainer is not None

    # 5. High-risk phishing feature vector
    malicious_vector = {
        "url_length": 140,
        "hostname_length": 45,
        "path_length": 65,
        "subdomain_depth": 3,
        "dot_count": 6,
        "hyphen_count": 4,
        "at_symbol_present": 1,
        "ip_in_hostname": 1,
        "digit_ratio": 0.25,
        "shannon_entropy": 5.1,
        "suspicious_tld_flag": 1,
        "brand_keyword_in_subdomain": 1,
        "punycode_flag": 0,
        "is_https": 0,
        "query_param_count": 4,
        "domain_age_days": 2,
        "ssl_valid": 0,
        "ssl_cert_age_days": -1,
        "ssl_is_self_signed": 1,
        "total_hops": 4,
        "has_shortener": 1,
        "redirect_loop_detected": 1,
        "has_password_input": 1,
        "external_form_action_count": 2,
        "hidden_iframe_count": 2,
        "external_script_ratio": 0.85,
        "obfuscated_js_heuristic": 1,
    }

    # 6. Benign legitimate feature vector
    benign_vector = {
        "url_length": 28,
        "hostname_length": 14,
        "path_length": 6,
        "subdomain_depth": 0,
        "dot_count": 1,
        "hyphen_count": 0,
        "at_symbol_present": 0,
        "ip_in_hostname": 0,
        "digit_ratio": 0.0,
        "shannon_entropy": 3.4,
        "suspicious_tld_flag": 0,
        "brand_keyword_in_subdomain": 0,
        "punycode_flag": 0,
        "is_https": 1,
        "query_param_count": 0,
        "domain_age_days": 4200,
        "ssl_valid": 1,
        "ssl_cert_age_days": 320,
        "ssl_is_self_signed": 0,
        "total_hops": 1,
        "has_shortener": 0,
        "redirect_loop_detected": 0,
        "has_password_input": 0,
        "external_form_action_count": 0,
        "hidden_iframe_count": 0,
        "external_script_ratio": 0.1,
        "obfuscated_js_heuristic": 0,
    }

    mal_res = classifier.predict(malicious_vector)
    ben_res = classifier.predict(benign_vector)

    # Assert probability ordering
    assert mal_res.phishing_probability > ben_res.phishing_probability
    assert mal_res.phishing_probability >= 0.70
    assert mal_res.risk_label == "HIGH"
    assert ben_res.risk_label in ("LOW", "MEDIUM")

    # Assert SHAP explanations
    assert len(mal_res.top_reasons) == 3
    for reason in mal_res.top_reasons:
        assert isinstance(reason.explanation, str)
        assert len(reason.explanation) > 10
        assert reason.direction in ("increases_risk", "decreases_risk")
        assert reason.feature_name in FEATURE_COLUMNS
