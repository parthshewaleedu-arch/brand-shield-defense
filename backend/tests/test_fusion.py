from __future__ import annotations

import pytest

from backend.app.fusion import MultiModalRiskFusion
from backend.app.schemas import (
    DomSignals,
    LexicalMlResult,
    RedirectChainResult,
    RedirectHop,
    VisualCvResult,
    VisualMatchCandidate,
)


@pytest.fixture
def fusion():
    return MultiModalRiskFusion()


@pytest.fixture
def clean_redirect_chain():
    return RedirectChainResult(
        original_url="https://example.com",
        final_url="https://example.com",
        registered_domain="example.com",
        hops=[
            RedirectHop(
                hop_number=1,
                url="https://example.com",
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


@pytest.fixture
def clean_dom_signals():
    return DomSignals(
        page_title="Example Domain",
        final_rendered_url="https://example.com",
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


@pytest.fixture
def benign_lexical_result():
    return LexicalMlResult(
        phishing_probability=0.05,
        risk_label="LOW",
        extracted_features={"url_length": 19},
        top_reasons=[],
        whois_lookup_failed=False,
        ssl_lookup_failed=False,
        model_version="test_v1",
    )


class TestHardOverrideRules:
    """Test suite for mandatory hard override rules."""

    def test_visual_brand_impersonation_override(
        self, fusion, clean_redirect_chain, clean_dom_signals, benign_lexical_result
    ):
        """Attacker hosts zero-day clone on clean cloud provider (e.g. vercel.app).

        URL lexical features appear benign (0.05 prob), but visual CV detects State Bank of India
        impersonation on an unauthorized domain. Must force CRITICAL_PHISHING.
        """
        visual_impersonation = VisualCvResult(
            matched_brand_id="sbi",
            matched_brand_name="State Bank of India",
            official_domains=["onlinesbi.sbi", "sbi.co.in"],
            cosine_similarity=0.91,
            phash_distance=2,
            combined_visual_score=0.92,
            domain_is_official=False,
            impersonation_detected=True,
            top_candidates=[
                VisualMatchCandidate(
                    brand_id="sbi",
                    brand_name="State Bank of India",
                    cosine_similarity=0.91,
                    phash_distance=2,
                    phash_similarity=0.96875,
                    combined_visual_score=0.92,
                )
            ],
            inference_time_ms=12,
            skipped_reason=None,
        )

        result = fusion.fuse(
            lexical_result=benign_lexical_result,
            visual_result=visual_impersonation,
            redirect_chain=clean_redirect_chain,
            dom_signals=clean_dom_signals,
            skip_sandbox=False,
        )

        assert result.verdict == "CRITICAL_PHISHING"
        assert result.final_risk_score >= 0.92
        assert result.risk_score_pct >= 92
        assert result.recommended_action == "BLOCK_CREDENTIALS"
        assert result.override_triggered is True
        assert result.override_reason is not None
        assert "State Bank of India" in result.override_reason
        assert "92%" in result.override_reason
        assert result.headline == result.override_reason

    def test_official_domain_safety_guard(
        self, fusion, clean_redirect_chain, clean_dom_signals
    ):
        """Authentic brand portal matches official domain.

        Even if lexical ML has elevated score (e.g. 0.40 due to short domain age or subdomains),
        official domain safety guard caps score <= 0.25 and marks SAFE.
        """
        elevated_lexical = LexicalMlResult(
            phishing_probability=0.40,
            risk_label="MEDIUM",
            extracted_features={},
            top_reasons=[],
            whois_lookup_failed=False,
            ssl_lookup_failed=False,
            model_version="test_v1",
        )

        official_visual = VisualCvResult(
            matched_brand_id="sbi",
            matched_brand_name="State Bank of India",
            official_domains=["onlinesbi.sbi", "sbi.co.in"],
            cosine_similarity=0.95,
            phash_distance=0,
            combined_visual_score=0.96,
            domain_is_official=True,
            impersonation_detected=False,
            top_candidates=[],
            inference_time_ms=10,
            skipped_reason=None,
        )

        result = fusion.fuse(
            lexical_result=elevated_lexical,
            visual_result=official_visual,
            redirect_chain=clean_redirect_chain,
            dom_signals=clean_dom_signals,
            skip_sandbox=False,
        )

        assert result.verdict == "SAFE"
        assert result.final_risk_score <= 0.25
        assert result.recommended_action == "ALLOW"
        assert "Verified authentic" in result.headline


class TestWeightedMultiModalFusion:
    """Test suite for weighted scoring and dynamic renormalization."""

    def test_skip_sandbox_renormalization(
        self, fusion, clean_redirect_chain, clean_dom_signals
    ):
        """When skip_sandbox=True, weights re-normalize to 0.78 * tabular + 0.22 * redirect_dom."""
        lexical = LexicalMlResult(
            phishing_probability=0.50,
            risk_label="MEDIUM",
            extracted_features={},
            top_reasons=[],
            whois_lookup_failed=False,
            ssl_lookup_failed=False,
            model_version="test_v1",
        )

        result = fusion.fuse(
            lexical_result=lexical,
            visual_result=None,
            redirect_chain=clean_redirect_chain,
            dom_signals=clean_dom_signals,
            skip_sandbox=True,
        )

        # clean redirect/dom score is 0.0
        # Expected: 0.78 * 0.50 = 0.39
        assert pytest.approx(result.final_risk_score, abs=0.01) == 0.39
        assert result.verdict == "SUSPICIOUS"
        assert result.recommended_action == "WARN"
        assert result.component_scores["visual_cv"] == 0.0

    def test_verdict_threshold_brackets(
        self, fusion, clean_redirect_chain, clean_dom_signals
    ):
        """Verify strict verdict mapping brackets (<0.35, 0.35-0.69, >=0.70)."""
        # 1. Low risk -> SAFE
        low_lex = LexicalMlResult(phishing_probability=0.10, risk_label="LOW", extracted_features={}, top_reasons=[], whois_lookup_failed=False, ssl_lookup_failed=False, model_version="v1")
        res_low = fusion.fuse(low_lex, None, clean_redirect_chain, clean_dom_signals, skip_sandbox=True)
        assert res_low.verdict == "SAFE"
        assert res_low.recommended_action == "ALLOW"

        # 2. Medium risk -> SUSPICIOUS
        med_lex = LexicalMlResult(phishing_probability=0.60, risk_label="MEDIUM", extracted_features={}, top_reasons=[], whois_lookup_failed=False, ssl_lookup_failed=False, model_version="v1")
        res_med = fusion.fuse(med_lex, None, clean_redirect_chain, clean_dom_signals, skip_sandbox=True)
        assert res_med.verdict == "SUSPICIOUS"
        assert res_med.recommended_action == "WARN"

        # 3. High risk -> CRITICAL_PHISHING
        high_lex = LexicalMlResult(phishing_probability=0.95, risk_label="HIGH", extracted_features={}, top_reasons=[], whois_lookup_failed=False, ssl_lookup_failed=False, model_version="v1")
        res_high = fusion.fuse(high_lex, None, clean_redirect_chain, clean_dom_signals, skip_sandbox=True)
        assert res_high.verdict == "CRITICAL_PHISHING"
        assert res_high.recommended_action == "BLOCK_CREDENTIALS"


class TestRedirectDomHeuristics:
    """Test suite for Phase 1 signals scoring."""

    def test_redirect_dom_scoring_accumulation(self, fusion):
        suspicious_chain = RedirectChainResult(
            original_url="https://bit.ly/test",
            final_url="https://evil.xyz",
            registered_domain="evil.xyz",
            hops=[],
            total_hops=4,  # +0.20
            has_shortener=True,  # +0.25
            redirect_loop_detected=True,  # +0.35
            errors=[],
        )

        suspicious_dom = DomSignals(
            page_title="Phish",
            final_rendered_url="https://evil.xyz",
            has_password_input=True,
            password_input_count=1,
            form_count=1,
            external_form_action_count=1,  # +0.35 (password + external action)
            suspicious_form_actions=["https://attacker-stealer.com/post"],
            iframe_count=1,
            hidden_iframe_count=1,  # +0.15
            script_count=5,
            external_script_ratio=0.8,
            favicon_url=None,
            inline_event_handler_count=2,
            obfuscated_js_heuristic=True,  # +0.20
        )

        score = fusion.compute_redirect_dom_score(suspicious_chain, suspicious_dom)
        # Sum exceeds 1.0, must clamp to 1.0
        assert score == 1.0
