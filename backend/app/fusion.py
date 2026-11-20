from __future__ import annotations

import logging
from typing import Optional

from .schemas import (
    DomSignals,
    FusionSummary,
    LexicalMlResult,
    RedirectChainResult,
    VisualCvResult,
)

logger = logging.getLogger("brandshield.fusion")


class MultiModalRiskFusion:
    """Stateless, deterministic multi-modal risk fusion engine.

    Synthesizes signals from:
    1. Tabular ML & SHAP lexical/infrastructure predictions (Phase 2)
    2. Computer Vision deep embedding & pHash brand impersonation (Phase 3)
    3. Headless browser redirect hops and DOM telemetry (Phase 1)
    """

    def compute_redirect_dom_score(
        self,
        redirect_chain: RedirectChainResult,
        dom_signals: Optional[DomSignals] = None,
    ) -> float:
        """Calculate a normalized redirect and DOM heuristic risk score (0.0 to 1.0)."""
        score = 0.0

        if redirect_chain.total_hops >= 3:
            score += 0.20
        if redirect_chain.has_shortener:
            score += 0.25
        if redirect_chain.redirect_loop_detected:
            score += 0.35

        if dom_signals is not None:
            # External form submission combined with password input is a critical phishing indicator
            if dom_signals.has_password_input and dom_signals.external_form_action_count > 0:
                score += 0.35
            elif dom_signals.external_form_action_count > 0:
                score += 0.20

            if dom_signals.hidden_iframe_count > 0:
                score += 0.15
            if dom_signals.obfuscated_js_heuristic:
                score += 0.20

        return float(min(1.0, max(0.0, score)))

    def fuse(
        self,
        lexical_result: LexicalMlResult,
        visual_result: Optional[VisualCvResult],
        redirect_chain: RedirectChainResult,
        dom_signals: Optional[DomSignals] = None,
        skip_sandbox: bool = False,
    ) -> FusionSummary:
        """Synthesize all subsystem results into a definitive risk verdict."""
        tabular_prob = float(lexical_result.phishing_probability)
        redirect_dom_score = self.compute_redirect_dom_score(redirect_chain, dom_signals)

        visual_active = (
            not skip_sandbox
            and visual_result is not None
            and visual_result.skipped_reason is None
        )

        # 1. Compute visual component risk
        visual_risk = 0.0
        if visual_active and visual_result is not None:
            if visual_result.domain_is_official:
                visual_risk = 0.0
            else:
                # Gate smoothly: background/incidental similarity below 0.55 contributes 0.0
                if visual_result.combined_visual_score >= 0.55:
                    visual_risk = float(visual_result.combined_visual_score)
                else:
                    visual_risk = 0.0

        # 2. Weighted multi-modal baseline score
        if visual_active:
            weighted_score = (
                0.60 * tabular_prob
                + 0.25 * visual_risk
                + 0.15 * redirect_dom_score
            )
        else:
            # Re-normalize weights when visual CV is bypassed or skipped
            weighted_score = (
                0.78 * tabular_prob
                + 0.22 * redirect_dom_score
            )

        weighted_score = float(min(1.0, max(0.0, weighted_score)))

        component_scores = {
            "tabular_ml": round(tabular_prob, 4),
            "visual_cv": round(visual_risk, 4),
            "redirect_dom": round(redirect_dom_score, 4),
        }

        # 3. Hard Rule 1: Visual Brand Impersonation Override (Mandatory per GEMINI.md)
        # Visual similarity on an unauthorized domain MUST force CRITICAL_PHISHING
        # regardless of whether lexical features appear benign.
        if visual_result is not None and visual_result.impersonation_detected:
            final_risk_score = max(weighted_score, 0.92)
            similarity_pct = int(round(visual_result.combined_visual_score * 100))
            brand = visual_result.matched_brand_name or visual_result.matched_brand_id or "Protected Brand"
            reason = f"Visual layout matches {brand} ({similarity_pct}% similarity) on an unverified domain"

            return FusionSummary(
                verdict="CRITICAL_PHISHING",
                final_risk_score=final_risk_score,
                risk_score_pct=int(round(final_risk_score * 100)),
                recommended_action="BLOCK_CREDENTIALS",
                override_triggered=True,
                override_reason=reason,
                component_scores=component_scores,
                headline=reason,
            )

        # 4. Hard Rule 2: Verified Official Domain Safety Guard
        # Authentic brand portals with clean redirects and no external credential-stealing forms
        # are guaranteed not to false-alarm.
        if (
            visual_result is not None
            and visual_result.domain_is_official
            and not redirect_chain.redirect_loop_detected
            and (dom_signals is None or dom_signals.external_form_action_count == 0)
        ):
            final_risk_score = min(weighted_score, 0.25)
            brand = visual_result.matched_brand_name or visual_result.matched_brand_id or "Official Brand"
            override_triggered = weighted_score > 0.25
            reason = f"Verified authentic domain for {brand}" if override_triggered else None

            return FusionSummary(
                verdict="SAFE",
                final_risk_score=final_risk_score,
                risk_score_pct=int(round(final_risk_score * 100)),
                recommended_action="ALLOW",
                override_triggered=override_triggered,
                override_reason=reason,
                component_scores=component_scores,
                headline=f"Verified authentic portal for {brand}.",
            )

        # 5. Standard Verdict Thresholds
        if weighted_score < 0.35:
            verdict = "SAFE"
            action = "ALLOW"
            headline = "No significant phishing or impersonation indicators detected."
        elif weighted_score < 0.70:
            verdict = "SUSPICIOUS"
            action = "WARN"
            headline = "Suspicious signals detected. Verify website identity before entering credentials."
        else:
            verdict = "CRITICAL_PHISHING"
            action = "BLOCK_CREDENTIALS"
            headline = "Critical phishing risk detected. Highly advised to avoid interacting with this website."

        return FusionSummary(
            verdict=verdict,
            final_risk_score=weighted_score,
            risk_score_pct=int(round(weighted_score * 100)),
            recommended_action=action,
            override_triggered=False,
            override_reason=None,
            component_scores=component_scores,
            headline=headline,
        )
