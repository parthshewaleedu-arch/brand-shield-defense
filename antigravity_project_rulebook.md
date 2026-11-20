# GEMINI.md — Project Architecture & Coding Standards

## 1. Project Identity
- **Project Name:** Multi-Modal Brand Impersonation & Visual Phishing Defense System (BrandShield AI)
- **Domain:** Cybersecurity / Applied Data Science / Multi-Modal Machine Learning
- **Academic Context:** SY B.Tech CSE (Data Science), MIT Academy of Engineering (AY 2026–27)
- **Team:** Aditya Badgujar, Parth Shewale, Gopal Rathod, Onkar Thakare

## 2. Core Technology Stack (STRICT ADHERENCE)
- **Language:** Python 3.11+
- **Backend API:** FastAPI (`fastapi`, `uvicorn`, `pydantic>=2.0`) — **NEVER** use Flask.
- **Headless Sandbox & Redirect Engine:** `playwright.async_api` (Chromium) — **NEVER** use Selenium or raw `requests` for page rendering or screenshot capture.
- **Tabular ML & Explainability:** `lightgbm` (primary) / `xgboost`, `scikit-learn`, `shap` (`TreeExplainer`), `python-whois`, `tldextract`, `cryptography` (for SSL inspection).
- **Computer Vision (Brand Impersonation):** `torch`, `torchvision` (`mobilenet_v3_small` pretrained weights), `imagehash` (`phash`), `Pillow`, `numpy`.
- **Browser Extension:** Chrome/Edge Extension Manifest V3 (`manifest.json` v3, `service_worker`, `content_scripts`). Zero remote hosted code.
- **Database / Storage:** SQLite (`aiosqlite`) for local scan logs & audit history; local filesystem (`data/brand_vault/`) for precomputed brand embedding tensors (`.pt` / `.npy`) and reference PNGs.
- **Testing & Verification:** `pytest`, `pytest-asyncio`, `httpx` (for FastAPI `AsyncClient` tests).

## 3. Architectural Mandates
1. **Asynchronous Execution:** The `/api/v1/analyze` pipeline MUST execute network-bound and browser-bound tasks cleanly using `asyncio` without blocking the FastAPI event loop.
2. **Sandbox Safety & Timeouts:**
   - Every Playwright browser context MUST disable file downloads (`accept_downloads=False`), ignore HTTPS errors (`ignore_https_errors=True` so expired/self-signed phishing certs still render for visual inspection), and enforce a strict `8000ms` navigation timeout.
   - Viewport MUST be locked to `1280x720` (`device_scale_factor=1`) so live screenshots match the reference `brand_vault` aspect ratio.
3. **Domain Normalization:** Always extract the registered root domain (eTLD+1) using `tldextract` (e.g., `login.sbi-verify.vercel.app` -> registered domain is `vercel.app`, subdomain is `login.sbi-verify`).
4. **Multi-Modal Override Rule:** If `visual_similarity >= 0.82` for a protected brand AND `registered_domain` is NOT in that brand's `official_domains` whitelist, the fusion engine MUST classify the target as `CRITICAL_PHISHING` regardless of whether lexical URL features appear clean.
5. **Explainability Mandate:** Every classification response MUST include the top 3 SHAP feature contributions translated into plain-English security warnings (e.g., *"Domain registered 2 days ago (+0.38 risk)"*).

## 4. Coding & File Hygiene Rules
- Type-hint all Python functions and Pydantic models strictly.
- Never hardcode absolute paths (e.g., `/home/user/...`); always resolve paths relative to `Path(__file__).resolve().parents[...]`.
- Do not commit large raw datasets (`*.csv` > 25MB) or binary model weights (`*.pkl`, `*.pt`) without updating `.gitignore`; provide deterministic generator/training scripts so any team member can reproduce artifacts locally.
- **Verification Loop:** Before concluding any task in Antigravity, run `pytest` via the terminal tool and confirm 100% pass rate.