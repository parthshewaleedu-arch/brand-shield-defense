# MASTER_PROMPT.md — Complete System Blueprint & Phased Build Guide

Use this document as your comprehensive specification. Feed the **Master Initialization Prompt** in Section 2 to Google Antigravity first to scaffold the repository, then execute **Phases 1 through 6** sequentially.

---

## 1. Complete Phased Building Process

| Phase | Milestone Name | Core Deliverables | Verification Gate (`pytest` / Demo) |
| :--- | :--- | :--- | :--- |
| **Phase 0** | **Repo Scaffolding & Brand Vault Seeding** | Directory tree, `pyproject.toml`/`requirements.txt`, `schemas.py`, `brand_vault.json` registry + automated script to capture clean reference screenshots for 15–20 brands. | `data/brand_vault/` populated with PNGs + embeddings; FastAPI `/health` returns `200 OK`. |
| **Phase 1** | **Redirect Unwrapper & Playwright Sandbox** | `unwrapper.py` (traces HTTP 301/302, `<meta refresh>`, JS redirects up to 10 hops) + `sandbox.py` (headless Chromium DOM extractor + `1280x720` screenshot buffer). | `test_sandbox.py` verifies redirect chain unwrapping and DOM signal extraction (`has_password_field`, `external_form_action`). |
| **Phase 2** | **Tabular Feature Engine, LightGBM & SHAP** | `lexical_ml.py` (25+ lexical, WHOIS, SSL, and DOM features), `train_tabular.py` (synthetic + Kaggle/OpenPhish trainer), and `shap.TreeExplainer` plain-English mapper. | `test_lexical_ml.py` confirms inference `< 50ms` and outputs 3 human-readable SHAP explanations. |
| **Phase 3** | **Visual Brand-Impersonation Engine (CV)** | `visual_cv.py` combining MobileNetV3-Small penultimate layer cosine similarity + `imagehash` perceptual hash (`pHash`) against `brand_vault`. | `test_visual_cv.py` detects a cloned HTML login page on `localhost` as >85% visual match to target brand. |
| **Phase 4** | **Multi-Modal Fusion Engine & FastAPI API** | `fusion.py` (weighted score combiner + Brand Impersonation hard override), SQLite scan history logger, and `POST /api/v1/analyze` endpoint. | `test_api.py` passes full end-to-end payload validation (`SAFE`, `SUSPICIOUS`, `CRITICAL_PHISHING`). |
| **Phase 5** | **Chrome Manifest V3 Extension (Prevention)** | `extension/` (`manifest.json`, `background.js`, `content.js`, `popup.html/js`) that intercepts navigation and blocks password input on phishing pages. | Live browser test: opening a spoofed bank login triggers full-screen block overlay + SHAP popup. |
| **Phase 6** | **SOC Web Dashboard & Review Demo Suite** | `dashboard/` web UI displaying live URL scanner, redirect hop visualizer, side-by-side visual comparison (Target vs. Official Brand), and SHAP bar chart. | Full offline/local demo script ready for SY Project Review presentation. |

---

## 2. Master Prompt for Google Antigravity (Copy & Paste into Agent Chat)

```text
You are the Lead Systems & ML Architect for the "Multi-Modal Brand Impersonation & Visual Phishing Defense System" (BrandShield AI). Read @GEMINI.md and @MEMORY.md first, and strictly adhere to all architectural rules defined there.

### SYSTEM OBJECTIVE
Build a production-grade, local-first, real-time phishing detection and prevention platform that defeats zero-day phishing attacks hosted on legitimate cloud infrastructure (e.g., Vercel, Netlify, Firebase, compromised WordPress) by fusing four core intelligence modules:
1. **Redirect Chain & Obfuscation Unwrapper (`backend/app/core/unwrapper.py`):** Follows multi-hop HTTP 301/302 redirects, URL shorteners, `<meta http-equiv="refresh">`, and JavaScript location redirects (up to 10 hops) to expose the true final landing URL and intermediate infrastructure.
2. **Headless Browser Sandbox & Visual Brand-Impersonation Detector (`backend/app/core/sandbox.py` & `backend/app/core/visual_cv.py`):** Renders the final landing page in an isolated headless Chromium instance (1280x720 viewport), captures a screenshot and DOM telemetry, and compares the visual layout against a protected Brand Vault (`data/brand_vault/`) using MobileNetV3-Small deep feature embeddings (cosine similarity) + Perceptual Hashing (pHash). If visual similarity >= 0.82 and the registered eTLD+1 domain does not belong to that brand's official domain list, trigger a CRITICAL Brand Impersonation alert.
3. **Tabular ML + SHAP Explainable AI Engine (`backend/app/core/lexical_ml.py` & `backend/train_tabular.py`):** Extracts 25+ features across URL lexical structure (entropy, length, IP usage, suspicious tokens, subdomain depth), domain intelligence (WHOIS domain age, SSL certificate validity/issuer), and DOM structure (password inputs, external form actions, hidden iframes, external script ratio). Classifies using LightGBM and computes top-3 SHAP feature contributions mapped to clear, human-readable explanations.
4. **Real-Time Browser Extension & SOC Dashboard (`extension/` & `dashboard/`):** A Chrome Manifest V3 extension that queries `POST http://localhost:8000/api/v1/analyze` on navigation, displays SHAP + visual similarity telemetry in the extension popup, and injects an active DOM warning shield (`content.js`) that locks `<input type="password">` fields on flagged pages.

### DATA CONTRACT (`backend/app/schemas.py`)
Every analysis request (`AnalyzeRequest(url: HttpUrl, force_rescan: bool = False)`) MUST return an `AnalyzeResponse` JSON containing:
- `request_id`: str (UUID)
- `initial_url`: str
- `final_url`: str
- `registered_domain`: str
- `redirect_chain`: list[dict] (`hop`, `url`, `status_code`, `redirect_type`)
- `verdict`: Literal["SAFE", "SUSPICIOUS", "CRITICAL_PHISHING"]
- `final_risk_score`: float (0.0 to 100.0)
- `tabular_ml_probability`: float (0.0 to 1.0)
- `visual_analysis`: dict (`matched_brand`: str | None, `cosine_similarity`: float, `phash_similarity`: float, `is_impersonation`: bool, `screenshot_b64`: str | None)
- `dom_signals`: dict (`has_password_field`: bool, `external_form_action`: bool, `iframe_count`: int, `page_title`: str)
- `shap_explanations`: list[dict] (top 3 items with `feature`, `value`, `impact_score`, `description`)
- `latency_ms`: float

### EXECUTION INSTRUCTIONS
Start with **Phase 0** only:
1. Explore the workspace and generate a clear Implementation Plan Artifact for Phase 0.
2. Scaffold the complete directory structure (`backend/`, `data/brand_vault/`, `extension/`, `dashboard/`).
3. Create `requirements.txt`, `.gitignore`, `backend/app/schemas.py`, `backend/app/main.py` (with `/health` and stubbed `/api/v1/analyze`), and `data/brand_vault/brands.json` (containing official domains and login URLs for 15 major brands: Google, Microsoft, Amazon, PayPal, SBI, HDFC Bank, ICICI Bank, Axis Bank, Paytm, PhonePe, GitHub, LinkedIn, Instagram, Netflix, and MITAOE Portal).
4. Create `backend/scripts/seed_brand_vault.py` that uses Playwright to capture clean baseline screenshots (or generates deterministic test fixtures if offline) and precomputes MobileNetV3 + pHash embeddings.
5. Write `backend/tests/test_health.py`, run `pytest`, and update `@MEMORY.md` with Phase 0 completion status before stopping for my review.
```

---

## 3. Individual Phase Follow-Up Prompts (Run After Phase 0 Passes)

### Prompt for Phase 1 (Redirect Unwrapper & Playwright Sandbox)
> "Read `@GEMINI.md` and `@MEMORY.md`. Plan and implement Phase 1: `backend/app/core/unwrapper.py` and `backend/app/core/sandbox.py`. Ensure `unwrapper.py` handles HTTP redirects, meta-refresh, and JS redirects safely up to 10 hops with loop detection. Ensure `sandbox.py` captures the 1280x720 screenshot and DOM telemetry (`has_password_field`, `external_form_action`, `iframe_count`, `external_resource_ratio`, `page_title`). Create `backend/tests/test_sandbox.py` using a local `pytest` HTTP test server fixture (so tests run 100% offline), verify all tests pass, and update `@MEMORY.md`."

### Prompt for Phase 2 (Tabular ML + SHAP Explainability)
> "Read `@GEMINI.md` and `@MEMORY.md`. Plan and implement Phase 2: `backend/app/core/lexical_ml.py` and `backend/train_tabular.py`. Implement 25+ lexical, SSL, WHOIS (with graceful offline fallback), and DOM feature extractors. Generate a realistic balanced training dataset generator in `train_tabular.py` (plus support for loading CSVs from `data/datasets/`) to train a `LGBMClassifier` saved to `data/models/lgbm_phish.joblib`. Implement `SHAPExplainer` returning the top 3 plain-English risk triggers. Write `backend/tests/test_lexical_ml.py`, run `pytest`, and update `@MEMORY.md`."

### Prompt for Phase 3 (Visual Brand-Impersonation Engine)
> "Read `@GEMINI.md` and `@MEMORY.md`. Plan and implement Phase 3: `backend/app/core/visual_cv.py`. Load `mobilenet_v3_small` (eval mode, CPU/CUDA auto-detect) to extract L2-normalized 576-dim feature vectors, combined with `imagehash.phash` similarity. Compare any incoming screenshot against `data/brand_vault/` embeddings and verify whether `registered_domain` belongs to the matched brand's `official_domains`. Write `backend/tests/test_visual_cv.py` testing both a legitimate brand domain and a spoofed lookalike domain, run `pytest`, and update `@MEMORY.md`."

### Prompt for Phase 4 (Multi-Modal Fusion & FastAPI Wiring)
> "Read `@GEMINI.md` and `@MEMORY.md`. Plan and implement Phase 4: `backend/app/fusion.py`, SQLite async history logging in `backend/app/database.py`, and wire everything into `backend/app/main.py` (`POST /api/v1/analyze` and `GET /api/v1/history`). Enforce the Brand Impersonation hard override rule from `@GEMINI.md`. Write full integration tests in `backend/tests/test_api.py`, run `pytest`, and update `@MEMORY.md`."

### Prompt for Phase 5 (Chrome Manifest V3 Extension)
> "Read `@GEMINI.md` and `@MEMORY.md`. Plan and implement Phase 5 inside `extension/`: `manifest.json` (MV3), `background.js` (caches recent URL verdicts for 5 minutes and queries `http://localhost:8000/api/v1/analyze`), `content.js` (injects a high-contrast blocking overlay and disables `<input type='password'>` fields when verdict is `CRITICAL_PHISHING` or `SUSPICIOUS`), and `popup.html` + `popup.js` (shows verdict badge, matched brand, visual similarity %, and top 3 SHAP reasons). Update `@MEMORY.md`."

### Prompt for Phase 6 (SOC Dashboard & Live Demo Spoof Simulator)
> "Read `@GEMINI.md` and `@MEMORY.md`. Plan and implement Phase 6: Build a modern dark-mode SOC Web Dashboard in `dashboard/` (served directly by FastAPI at `/`) allowing manual URL scans, viewing recent extension blocks from SQLite, visualizing the redirect hop chain, displaying SHAP feature contribution bars, and comparing the suspect screenshot side-by-side against the matched official brand screenshot. Also add `backend/scripts/demo_spoof_server.py` (a local mock phishing page mimicking a bank login on `localhost:9090`) so we can demonstrate real-time detection and extension blocking offline during our college project review. Update `@MEMORY.md`."