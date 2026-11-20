#!/usr/bin/env python3
"""BrandShield AI - Interactive Demo Scanner CLI."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


GREEN = "92"
YELLOW = "93"
RED = "91"
BLUE = "94"
CYAN = "96"
BOLD = "1"
DIM = "2"


def main():
    parser = argparse.ArgumentParser(description="BrandShield AI - Demo URL Scanner")
    parser.add_argument("url", nargs="?", default="https://example.com", help="Target URL to inspect")
    parser.add_argument("--skip-sandbox", action="store_true", help="Bypass headless browser sandbox rendering")
    parser.add_argument("--include-screenshot", action="store_true", help="Include base64 screenshot in output")
    parser.add_argument("--host", default="http://127.0.0.1:8000", help="Backend API server base URL")

    args = parser.parse_args()

    api_url = f"{args.host}/api/v1/analyze"

    print(color("\n" + "=" * 60, BOLD))
    print(color("  🛡️  BrandShield AI — Visual Phishing Defense System", BOLD + ";" + CYAN))
    print(color("=" * 60 + "\n", BOLD))

    payload = {
        "url": args.url,
        "skip_sandbox": args.skip_sandbox,
        "include_screenshot": args.include_screenshot,
    }

    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "BrandShieldDemo/1.0"},
    )

    print(f"[*] Submitting scan target: {color(args.url, BOLD)}")
    print(f"[*] Querying API endpoint:  {api_url}")
    print("[*] Performing multi-hop unwrap, DOM extraction, Tabular ML & Computer Vision...\n")

    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(color(f"[!] Connection failed: {e}", RED))
        print(color(f"\nMake sure the backend server is running:", YELLOW))
        print(color("   uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload\n", BOLD))
        sys.exit(1)

    fusion = data["fusion"]
    verdict = fusion["verdict"]
    score_pct = fusion["risk_score_pct"]
    action = fusion["recommended_action"]

    if verdict == "SAFE":
        badge = color(f"  [ SAFE ]  ", "42;30;" + BOLD)
        score_color = GREEN
    elif verdict == "SUSPICIOUS":
        badge = color(f" [ SUSPICIOUS ] ", "43;30;" + BOLD)
        score_color = YELLOW
    else:
        badge = color(f" [ CRITICAL_PHISHING ] ", "41;97;" + BOLD)
        score_color = RED

    print(f"VERDICT:              {badge}")
    print(f"RISK SCORE:           {color(f'{score_pct}%', score_color + ';' + BOLD)} ({fusion['final_risk_score']:.4f})")
    print(f"RECOMMENDED ACTION:   {color(action, BOLD)}")
    print(f"HEADLINE:             {color(fusion['headline'], BOLD)}")

    if fusion.get("override_triggered"):
        print(f"OVERRIDE TRIGGERED:   {color('YES (Hard Brand Impersonation Override)', RED + ';' + BOLD)}")
        print(f"OVERRIDE REASON:      {color(fusion['override_reason'], RED)}")

    print("\n" + color("--- Subsystem Breakdown ---", CYAN))
    comp = fusion["component_scores"]
    print(f"• Tabular ML Score:   {comp.get('tabular_ml', 0.0):.4f}")
    print(f"• Visual CV Risk:     {comp.get('visual_cv', 0.0):.4f}")
    print(f"• Redirect/DOM Score: {comp.get('redirect_dom', 0.0):.4f}")

    # Redirect info
    rc = data.get("redirect_chain", {})
    print(f"\n" + color("--- Redirect Chain ---", CYAN))
    print(f"• Original URL:       {rc.get('original_url')}")
    print(f"• Final Landing URL:  {rc.get('final_url')}")
    print(f"• Registered Domain:  {rc.get('registered_domain')}")
    print(f"• Total Hops:         {rc.get('total_hops')}")

    # Visual CV details
    vcv = data.get("visual_cv")
    if vcv:
        print(f"\n" + color("--- Computer Vision Brand Impersonation ---", CYAN))
        print(f"• Matched Brand:      {vcv.get('matched_brand_name') or 'None'}")
        print(f"• Cosine Similarity:  {vcv.get('cosine_similarity', 0.0):.4f}")
        print(f"• pHash Distance:     {vcv.get('phash_distance')}")
        print(f"• Combined Score:     {vcv.get('combined_visual_score', 0.0):.4f}")
        print(f"• Domain Official:    {vcv.get('domain_is_official')}")
        print(f"• Impersonation Flag: {color(str(vcv.get('impersonation_detected')), RED if vcv.get('impersonation_detected') else GREEN)}")

    # SHAP explainability
    lml = data.get("lexical_ml", {})
    reasons = lml.get("top_reasons", [])
    if reasons:
        print(f"\n" + color("--- Top SHAP Explainability Triggers ---", CYAN))
        for i, r in enumerate(reasons, 1):
            impact_sign = "+" if r["shap_impact"] > 0 else ""
            print(f"  {i}. {r['explanation']} ({impact_sign}{r['shap_impact']:.3f})")

    print(f"\n" + color(f"Processing Latency: {data.get('processing_time_ms', 0)} ms | Cached: {data.get('cached')}", DIM))
    print(color("=" * 60 + "\n", BOLD))


if __name__ == "__main__":
    main()
