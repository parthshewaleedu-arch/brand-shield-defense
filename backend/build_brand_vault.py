from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import imagehash
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Set up project path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.schemas import BrandRecord
from backend.app.core.visual_cv import MobileNetV3Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_brand_vault")

DATA_DIR = PROJECT_ROOT / "data" / "brand_vault"
IMAGES_DIR = DATA_DIR / "images"
BRANDS_JSON = DATA_DIR / "brands.json"
MODELS_DIR = PROJECT_ROOT / "backend" / "models"
INDEX_FILE = MODELS_DIR / "brand_vault_index.npz"
META_FILE = MODELS_DIR / "brand_vault_meta.json"

BRAND_PALETTES = {
    "sbi": {"bg": (240, 244, 248), "primary": (0, 51, 102), "accent": (0, 153, 204), "title": "STATE BANK OF INDIA - NET BANKING"},
    "hdfc": {"bg": (245, 247, 250), "primary": (0, 76, 143), "accent": (237, 35, 42), "title": "HDFC BANK - SECURE NETBANKING"},
    "icici": {"bg": (255, 248, 240), "primary": (163, 32, 32), "accent": (243, 112, 33), "title": "ICICI BANK - INTERNET BANKING"},
    "axis": {"bg": (250, 242, 245), "primary": (151, 20, 77), "accent": (180, 40, 90), "title": "AXIS BANK - CORPORATE & RETAIL LOGIN"},
    "paytm": {"bg": (240, 248, 255), "primary": (15, 74, 138), "accent": (0, 185, 245), "title": "PAYTM - SECURE PAYMENT LOGIN"},
    "phonepe": {"bg": (245, 240, 250), "primary": (95, 37, 159), "accent": (130, 60, 200), "title": "PHONEPE - UPI & WALLET PORTAL"},
    "google": {"bg": (255, 255, 255), "primary": (66, 133, 244), "accent": (52, 168, 83), "title": "Google Sign In - Accounts"},
    "microsoft": {"bg": (244, 244, 244), "primary": (0, 103, 184), "accent": (0, 164, 239), "title": "Microsoft - Sign In to your Account"},
    "amazon": {"bg": (245, 245, 245), "primary": (19, 25, 33), "accent": (240, 193, 75), "title": "Amazon.com - Secure Sign In"},
    "apple": {"bg": (245, 245, 247), "primary": (29, 29, 31), "accent": (0, 113, 227), "title": "Apple ID - Sign In with Apple"},
    "github": {"bg": (13, 17, 23), "primary": (22, 27, 34), "accent": (46, 160, 67), "title": "GitHub - Sign in to GitHub"},
    "instagram": {"bg": (250, 250, 250), "primary": (193, 53, 132), "accent": (225, 48, 108), "title": "Instagram - Member Login"},
    "linkedin": {"bg": (243, 242, 239), "primary": (10, 102, 194), "accent": (0, 65, 130), "title": "LinkedIn - Welcome Back"},
    "paypal": {"bg": (245, 247, 250), "primary": (0, 48, 135), "accent": (0, 121, 193), "title": "PayPal - Checkout & Portal"},
    "mitaoe": {"bg": (240, 245, 250), "primary": (27, 54, 93), "accent": (217, 130, 43), "title": "MIT ACADEMY OF ENGINEERING - STUDENT PORTAL"},
}


def draw_synthetic_screenshot(brand_id: str, brand_name: str, width: int = 1280, height: int = 720) -> Image.Image:
    """Generate a realistic, deterministic 1280x720 mock screenshot of a brand's login/portal."""
    palette = BRAND_PALETTES.get(brand_id, {
        "bg": (245, 245, 245),
        "primary": (50, 50, 80),
        "accent": (70, 130, 180),
        "title": f"{brand_name.upper()} LOGIN",
    })

    img = Image.new("RGB", (width, height), palette["bg"])
    draw = ImageDraw.Draw(img)

    # 1. Header bar
    header_height = 80
    draw.rectangle([0, 0, width, header_height], fill=palette["primary"])
    draw.rectangle([0, header_height, width, header_height + 4], fill=palette["accent"])

    # Header title
    title_text = palette["title"]
    draw.text((40, 28), title_text, fill=(255, 255, 255))
    draw.text((width - 200, 32), "Official Portal", fill=(200, 220, 255))

    # 2. Central Login Card
    card_w = 460
    card_h = 440
    card_x = (width - card_w) // 2
    card_y = header_height + 60

    card_bg = (22, 27, 34) if brand_id == "github" else (255, 255, 255)
    text_color = (230, 230, 230) if brand_id == "github" else (40, 40, 40)
    input_bg = (13, 17, 23) if brand_id == "github" else (248, 250, 252)
    border_color = (48, 54, 61) if brand_id == "github" else (209, 213, 219)

    # Drop shadow simulation
    draw.rectangle([card_x + 3, card_y + 3, card_x + card_w + 3, card_y + card_h + 3], fill=(210, 215, 220))
    # Main card
    draw.rectangle([card_x, card_y, card_x + card_w, card_y + card_h], fill=card_bg, outline=border_color, width=2)

    # Card Brand Heading
    draw.text((card_x + 30, card_y + 35), f"{brand_name}", fill=palette["accent"])
    draw.text((card_x + 30, card_y + 65), "Enter your credentials to continue securely", fill=(120, 130, 140))

    # Input 1: Username / ID
    draw.text((card_x + 30, card_y + 115), "Username / Email ID", fill=text_color)
    draw.rectangle([card_x + 30, card_y + 140, card_x + card_w - 30, card_y + 185], fill=input_bg, outline=border_color, width=1)
    draw.text((card_x + 45, card_y + 155), "user@example.com", fill=(160, 170, 180))

    # Input 2: Password
    draw.text((card_x + 30, card_y + 205), "Password / Security PIN", fill=text_color)
    draw.rectangle([card_x + 30, card_y + 230, card_x + card_w - 30, card_y + 275], fill=input_bg, outline=border_color, width=1)
    draw.text((card_x + 45, card_y + 245), "••••••••••••", fill=(160, 170, 180))

    # Login Button
    button_y = card_y + 305
    btn_color = palette["accent"] if brand_id not in ("amazon", "google") else palette["primary"]
    btn_text_color = (255, 255, 255) if brand_id != "amazon" else (17, 17, 17)
    if brand_id == "amazon":
        btn_color = palette["accent"]
    draw.rectangle([card_x + 30, button_y, card_x + card_w - 30, button_y + 48], fill=btn_color)
    draw.text((card_x + 180, button_y + 16), "SIGN IN", fill=btn_text_color)

    # Security Badge / Footer
    draw.text((card_x + 65, card_y + 390), "🔒 256-bit TLS Protected Authentic Session", fill=(130, 140, 150))

    # 3. Page Footer
    draw.rectangle([0, height - 40, width, height], fill=(235, 238, 242))
    draw.text((width // 2 - 160, height - 26), f"© 2026 {brand_name}. All rights reserved.", fill=(110, 120, 130))

    return img


def bootstrap_vault() -> None:
    """Generate synthetic reference images for any missing brands in brands.json."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    if not BRANDS_JSON.exists():
        logger.error("brands.json not found at %s", BRANDS_JSON)
        return

    with open(BRANDS_JSON, "r", encoding="utf-8") as f:
        brands_data = json.load(f)

    logger.info("Bootstrapping reference images for %d brands...", len(brands_data))

    generated_count = 0
    for b in brands_data:
        brand_id = b["brand_id"]
        brand_name = b["brand_name"]
        ref_images = b.get("reference_images", [f"{brand_id}_login.png"])

        for ref_img_name in ref_images:
            img_path = IMAGES_DIR / ref_img_name
            if not img_path.exists():
                logger.info("Generating synthetic reference screenshot for %s -> %s", brand_name, img_path.name)
                mock_img = draw_synthetic_screenshot(brand_id, brand_name)
                mock_img.save(img_path, format="PNG")
                generated_count += 1
            else:
                logger.info("Existing reference image found for %s: %s", brand_id, img_path.name)

    logger.info("Bootstrap complete: generated %d reference images.", generated_count)


def build_vault_index() -> None:
    """Compute MobileNetV3 deep embeddings and perceptual hashes for all reference images

    and serialize the index and metadata to disk.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not BRANDS_JSON.exists():
        logger.error("brands.json does not exist at %s", BRANDS_JSON)
        return

    with open(BRANDS_JSON, "r", encoding="utf-8") as f:
        brands_raw = json.load(f)

    logger.info("Initializing MobileNetV3Embedder for Brand Vault indexing...")
    embedder = MobileNetV3Embedder()

    embeddings_list: list[np.ndarray] = []
    brand_ids_list: list[str] = []
    phashes_list: list[str] = []
    processed_brands: list[dict] = []

    for b_dict in brands_raw:
        brand_record = BrandRecord(**b_dict)
        brand_id = brand_record.brand_id
        valid_refs = []

        for ref_name in brand_record.reference_images:
            ref_path = IMAGES_DIR / ref_name
            if not ref_path.exists():
                logger.warning("Reference image %s for %s does not exist; skipping", ref_name, brand_id)
                continue

            try:
                with Image.open(ref_path) as img:
                    # 1. Compute 576-dim L2-normalized embedding
                    emb = embedder.extract_from_image(img)
                    embeddings_list.append(emb)

                    # 2. Compute 64-bit perceptual hash (stored as hex string)
                    ph = imagehash.phash(img)
                    phashes_list.append(str(ph))

                    brand_ids_list.append(brand_id)
                    valid_refs.append(ref_name)
            except Exception as e:
                logger.error("Error processing %s for %s: %s", ref_path, brand_id, e)

        if valid_refs:
            record_dict = brand_record.model_dump()
            record_dict["reference_images"] = valid_refs
            processed_brands.append(record_dict)

    if not embeddings_list:
        logger.error("No valid embeddings extracted. Brand Vault index not created.")
        return

    embeddings_arr = np.vstack(embeddings_list).astype(np.float32)
    brand_ids_arr = np.array(brand_ids_list, dtype=object)
    phashes_arr = np.array(phashes_list, dtype=object)

    logger.info("Saving Brand Vault index (%d entries, shape %s) to %s", len(embeddings_list), embeddings_arr.shape, INDEX_FILE)
    np.savez_compressed(
        INDEX_FILE,
        embeddings=embeddings_arr,
        brand_ids=brand_ids_arr,
        phashes=phashes_arr,
    )

    logger.info("Saving Brand Vault metadata (%d brands) to %s", len(processed_brands), META_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_brands, f, indent=2)

    logger.info("Brand Vault build completed successfully!")


async def capture_brand_screenshots() -> None:
    """Optional online capture using BrowserSandbox to capture official landing pages."""
    from backend.app.core.sandbox import BrowserSandbox

    with open(BRANDS_JSON, "r", encoding="utf-8") as f:
        brands_data = json.load(f)

    sandbox = BrowserSandbox()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for b in brands_data:
        brand_id = b["brand_id"]
        domains = b.get("official_domains", [])
        if not domains:
            continue
        primary_domain = domains[0]
        target_url = f"https://{primary_domain}"
        out_name = f"{brand_id}_login.png"
        out_path = IMAGES_DIR / out_name

        logger.info("Capturing live screenshot for %s from %s...", brand_id, target_url)
        try:
            result = await sandbox.render_and_extract(target_url)
            if result.success and result.screenshot_bytes_b64:
                import base64
                raw_bytes = base64.b64decode(result.screenshot_bytes_b64)
                with open(out_path, "wb") as f:
                    f.write(raw_bytes)
                logger.info("Saved screenshot to %s", out_path)
            else:
                logger.warning("Sandbox failed for %s: %s", target_url, result.error)
        except Exception as e:
            logger.error("Capture failed for %s: %s", target_url, e)


def main():
    parser = argparse.ArgumentParser(description="Brand Vault Builder for Visual Phishing Defense")
    parser.add_argument("--bootstrap", action="store_true", help="Generate synthetic reference screenshots and build index")
    parser.add_argument("--build", action="store_true", help="Build index from existing reference images")
    parser.add_argument("--capture", action="store_true", help="Capture live screenshots from official domains")

    args = parser.parse_args()

    if not (args.bootstrap or args.build or args.capture):
        parser.print_help()
        sys.exit(1)

    if args.capture:
        logger.info("Running online capture...")
        asyncio.run(capture_brand_screenshots())

    if args.bootstrap:
        logger.info("Bootstrapping reference images...")
        bootstrap_vault()
        logger.info("Building Brand Vault index...")
        build_vault_index()
    elif args.build:
        logger.info("Building Brand Vault index from existing images...")
        build_vault_index()


if __name__ == "__main__":
    main()
