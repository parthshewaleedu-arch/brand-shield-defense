# MEMORY.md — Persistent Project State & Architectural Log

> **AGENT INSTRUCTION:** Read this file at the start of every session. Update the **Current Phase Status**, **Completed Modules**, **Key Technical Decisions**, and **Known Issues / Next Action** sections before completing any prompt.

## 1. Current Phase Status
- **Active Phase:** `Phase 3 — Completed`
- **Overall Completion:** `55%`
- **Last Verified Test Run:** `47 passed in 21.58s (pytest -v: 100% pass rate, 0 failures, 0 warnings)`

## 2. Module Implementation Checklist
- [x] **Phase 0/1 Schemas:** `backend/app/schemas.py` (`AnalyzeRequest`, `RedirectHop`, `RedirectChainResult`, `DomSignals`, `SandboxResult`, `ShapReason`, `LexicalMlResult`, `BrandRecord`, `VisualMatchCandidate`, `VisualCvResult`)
- [x] **Phase 1:** `backend/app/core/unwrapper.py` (Multi-hop HTTP/Meta/JS redirect tracer with loop detection & shortener detector)
- [x] **Phase 1:** `backend/app/core/sandbox.py` (Headless Playwright Chromium DOM + 1280x720 viewport screenshot capture)
- [x] **Phase 2:** `backend/app/core/lexical_ml.py` + `backend/train_tabular.py` (27 canonical features, LightGBM, SHAP top-3 explainer)
- [x] **Phase 3:** `backend/app/core/visual_cv.py` + `backend/build_brand_vault.py` (MobileNetV3-Small embeddings + pHash brand impersonation engine & Brand Vault)
- [ ] **Phase 4:** `backend/app/fusion.py` + `backend/app/database.py` + `POST /api/v1/analyze`
- [ ] **Phase 5:** `extension/` (Chrome Manifest V3 real-time interceptor + password-field DOM blocker + SHAP popup)
- [ ] **Phase 6:** `dashboard/` (SOC Analytics Web UI) + `backend/scripts/demo_spoof_server.py`

## 3. Key Technical Decisions & Thresholds
- **Canonical Feature Ordering (27 Features):** Shared strictly between `train_tabular.py` and `lexical_ml.py` to prevent any training-serving skew:
  - 15 URL Lexical: `url_length`, `hostname_length`, `path_length`, `subdomain_depth`, `dot_count`, `hyphen_count`, `at_symbol_present`, `ip_in_hostname`, `digit_ratio`, `shannon_entropy`, `suspicious_tld_flag`, `brand_keyword_in_subdomain`, `punycode_flag`, `is_https`, `query_param_count`.
  - 4 Domain & SSL: `domain_age_days`, `ssl_valid`, `ssl_cert_age_days`, `ssl_is_self_signed` (asynchronous lookups with 3.0s timeouts via `asyncio.to_thread` and offline `-1` fallback).
  - 8 Phase 1 Signals: `total_hops`, `has_shortener`, `redirect_loop_detected`, `has_password_input`, `external_form_action_count`, `hidden_iframe_count`, `external_script_ratio`, `obfuscated_js_heuristic`.
- **SHAP TreeExplainer Normalization:** `LexicalPhishingClassifier` dynamically normalizes SHAP output structures into a consistent 1D positive-class array, selects top 3 drivers, and formats them into plain-English security warnings via `EXPLANATION_TEMPLATES`.
- **Visual Brand Impersonation Decision Rule:**
  - Combined visual score formula: `0.75 * cosine_similarity + 0.25 * phash_similarity`
  - Impersonation trigger: `not domain_is_official` AND `(cosine_similarity >= 0.82 or (cosine_similarity >= 0.75 and phash_distance <= 10))`
  - Domain ownership verification via `tldextract` strictly validating root domain (eTLD+1) and authorized subdomains against canonical brand official domains.
  - Blank/uniform page detection threshold: `np.std(gray) < 5.0` automatically skips uniform screens.
- **Model Champion Selection:** Prioritizes lowest False Negative Rate (FNR) with Precision >= 0.85 (breaking ties by PR-AUC). LightGBM selected with Accuracy: 1.0, Precision: 1.0, Recall: 1.0, F1: 1.0, PR-AUC: 1.0, FNR: 0.0000 on bootstrap benchmark.
- **Offline Resilience:** All tests use `unittest.mock` for WHOIS/SSL network calls and local HTTP fixtures for end-to-end sandbox/unwrapper verification.

## 4. Environment & Setup Notes
- **OS:** Linux (Ubuntu) / Python 3.14+
- **Playwright Browser Binary:** Chromium installed in `~/.cache/ms-playwright/chromium-1255/chrome-linux64/chrome`.
- **Installed Packages:** `fastapi`, `uvicorn`, `pydantic>=2.0`, `httpx`, `playwright`, `tldextract`, `pillow`, `pytest`, `pytest-asyncio`, `lightgbm`, `shap`, `scikit-learn`, `joblib`, `python-whois`, `pandas`, `imagehash`.

## 5. Session Log & Next Action
- **[Phase 1 Complete]:** Implemented unwrapper and sandbox pipeline (14 tests passed).
- **[Phase 2 Complete]:** Implemented 27-feature extractor, trained and benchmarked LightGBM champion (`tabular_model.joblib`), integrated `shap.TreeExplainer` with plain-English warnings, and validated 11 new tests (25 total tests passing).
- **[Phase 3 Complete]:** Implemented `BrandRecord`, `VisualMatchCandidate`, and `VisualCvResult` schemas; created `MobileNetV3Embedder`, `is_official_domain`, and `BrandVisualMatcher`; created `build_brand_vault.py` with 15 high-target brands and generated synthetic reference images; created test suite with 20 passing tests (45 total tests passing).
- **Immediate Next Action:** Proceed to **Phase 4: Multi-Modal Fusion Engine, Audit Database & FastAPI Analysis API** (`backend/app/fusion.py`, `backend/app/database.py`, `POST /api/v1/analyze`).
