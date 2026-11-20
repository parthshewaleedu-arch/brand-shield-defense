# WORKFLOW.md — Google Antigravity & Team Development Playbook

## 1. Local Setup Before Opening Google Antigravity
Run these commands once in your Ubuntu terminal to initialize the workspace cleanly:

```bash
mkdir -p brand-shield-defense && cd brand-shield-defense
git init
python3 -m venv .venv
source .venv/bin/activate

# Place GEMINI.md, MASTER_PROMPT.md, MEMORY.md, and WORKFLOW.md in this root folder
git add GEMINI.md MASTER_PROMPT.md MEMORY.md WORKFLOW.md
git commit -m "chore: initialize project governance, master prompt, and memory files"
```

---

## 2. How to Operate Google Antigravity Effectively

1. **Always Use `Planning` Mode (Agent-Assisted):**
   - Set Antigravity’s execution mode to **Planning / Agent-Assisted** (not blind auto-pilot) for Phases 0–6.
   - When Antigravity generates an **Implementation Plan Artifact**, inspect the file list and schema changes first. Leave inline comments directly on the plan if a library or path deviates from `GEMINI.md`, then click **Proceed**.
2. **One Phase Per Agent Session:**
   - Do not ask Antigravity to build all 6 phases in a single prompt.
   - Run one phase prompt $\rightarrow$ inspect the generated Walkthrough Artifact $\rightarrow$ verify `pytest` passes in the integrated terminal $\rightarrow$ `git commit` $\rightarrow$ start a fresh agent thread referencing `@MEMORY.md` for the next phase.
3. **Browser Subagent Verification (Phases 5 & 6):**
   - When testing the SOC Dashboard (`localhost:8000`) and the mock phishing server (`localhost:9090`), let Antigravity’s built-in Browser Subagent verify the UI layout and screenshot rendering automatically.

---

## 3. Parallel Team Work Distribution (4 Members)

Once **Phase 0** is committed to `main`, branch out so all four members contribute verifiable code to the repository:

| Team Member | Git Branch | Assigned Modules | Key Responsibilities |
| :--- | :--- | :--- | :--- |
| **Aditya Badgujar** | `feat/cv-fusion-api` | **Phase 3 & 4** (`visual_cv.py`, `fusion.py`, `main.py`) | MobileNetV3 + pHash brand similarity engine, multi-modal score fusion, FastAPI orchestration. |
| **Parth Shewale** | `feat/sandbox-unwrapper` | **Phase 1** (`unwrapper.py`, `sandbox.py`) | Multi-hop redirect chain tracer, Playwright headless sandbox, DOM telemetry extractor. |
| **Gopal Rathod** | `feat/tabular-shap` | **Phase 2** (`lexical_ml.py`, `train_tabular.py`) | 25+ URL/SSL/WHOIS features, LightGBM model training on Kaggle/OpenPhish, SHAP plain-English explainer. |
| **Onkar Thakare** | `feat/extension-dashboard` | **Phase 5 & 6** (`extension/`, `dashboard/`, `demo_spoof_server.py`) | Chrome Manifest V3 real-time blocker extension, SOC dashboard UI, and local spoofed bank demo setup. |

### Git Merge Protocol
```bash
# Before starting work on any module
git checkout main && git pull
git checkout -b feat/<your-module>

# After Antigravity completes the phase and pytest passes
pytest -v
git add .
git commit -m "feat(<module>): implement and verify <phase-description>"
git push -u origin feat/<your-module>
```

---

## 4. Live College Review Demo Script (5-Minute Walkthrough)
When presenting to **Mrs. Aparna Kulkarni** and the evaluation panel:
1. **Start Backend & Mock Target:** Run `uvicorn backend.app.main:app --port 8000` and `python backend/scripts/demo_spoof_server.py` (serves a fake SBI/HDFC login clone on `http://localhost:9090`).
2. **Demonstrate Active Prevention (Browser Extension):** Open `http://localhost:9090` in Chrome with the BrandShield MV3 extension active. Show the extension automatically intercepting the page, locking the password field, and popping up the warning banner: *"CRITICAL PHISHING: 94.2% Visual Match to Official Bank UI on Unverified Domain"*.
3. **Demonstrate Explainable AI (SHAP) & Redirect Unwrapping:** Open the SOC Dashboard (`http://localhost:8000`), paste a shortened/redirected link, and show the live redirect hop graph, side-by-side visual screenshot comparison, and top 3 SHAP risk contributors.