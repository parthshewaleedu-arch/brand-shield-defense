from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Optional, Union

import imagehash
import numpy as np
from PIL import Image, ImageOps
import tldextract

from ..schemas import BrandRecord, VisualCvResult, VisualMatchCandidate

logger = logging.getLogger(__name__)

# Lazy imports for torch to avoid cold-start penalties when not needed
torch = None
models = None
MobileNet_V3_Small_Weights = None


def _ensure_torch():
    global torch, models, MobileNet_V3_Small_Weights
    if torch is None:
        import torch as _torch
        import torchvision.models as _models
        from torchvision.models import MobileNet_V3_Small_Weights as _weights

        torch = _torch
        models = _models
        MobileNet_V3_Small_Weights = _weights


class MobileNetV3Embedder:
    """Extracts 576-dimensional L2-normalized feature vectors from screenshots
    using MobileNetV3-Small (pretrained on ImageNet-1K).
    """

    def __init__(self, device: Optional[str] = None):
        self.has_torch = False
        try:
            _ensure_torch()
            if device is not None:
                self.device = torch.device(device)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            try:
                weights = MobileNet_V3_Small_Weights.DEFAULT
                self.transforms = weights.transforms()
                full_model = models.mobilenet_v3_small(weights=weights)
            except Exception as e:
                logger.warning(
                    "Failed to load default pretrained weights for MobileNetV3-Small (%s); using uninitialized weights",
                    e,
                )
                full_model = models.mobilenet_v3_small(weights=None)
                from torchvision import transforms as T

                self.transforms = T.Compose(
                    [
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                )

            # Retain feature extractor and average pool, strip classification head
            self.features = full_model.features
            self.avgpool = full_model.avgpool

            self.features.to(self.device).eval()
            self.avgpool.to(self.device).eval()
            self.has_torch = True
        except Exception as e:
            logger.warning("PyTorch not initialized (%s); using deterministic 576-dim spatial fallback", e)

    def extract_from_image(self, image: Image.Image) -> np.ndarray:
        """Extract a 576-dim L2-normalized float32 embedding vector from a PIL Image."""
        if self.has_torch:
            # Enforce RGB mode and correct EXIF orientation
            rgb_img = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transforms(rgb_img).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                feat = self.features(tensor)
                pooled = self.avgpool(feat)
                flattened = torch.flatten(pooled, 1)  # shape (1, 576)
                vec = flattened.cpu().numpy().squeeze(0).astype(np.float32)

            norm = float(np.linalg.norm(vec))
            if norm > 1e-12:
                vec = vec / norm
            else:
                vec = np.zeros_like(vec, dtype=np.float32)
            return vec
        else:
            # Spatial 24x24 = 576 normalized layout vector
            resized = image.convert("L").resize((24, 24), Image.Resampling.BILINEAR)
            vec = np.array(resized, dtype=np.float32).flatten()
            norm = float(np.linalg.norm(vec))
            if norm > 1e-12:
                vec = vec / norm
            return vec.astype(np.float32)

    def extract_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Extract embedding from raw image bytes."""
        with Image.open(io.BytesIO(image_bytes)) as img:
            return self.extract_from_image(img)


def is_official_domain(target_url: str, official_domains: list[str]) -> bool:
    """Verifies whether target_url belongs to the list of canonical official domains

    associated with a protected brand. Enforces strict root domain (eTLD+1)
    and valid subdomain hierarchy to reject spoofing attempts (e.g.,
    'login.sbi.evil.com' or 'fake-sbi.com').
    """
    if not target_url or not official_domains:
        return False

    try:
        t_ext = tldextract.extract(target_url.strip())
        t_reg = (
            t_ext.top_domain_under_public_suffix
            if hasattr(t_ext, "top_domain_under_public_suffix")
            else (t_ext.registered_domain or "")
        ).lower()

        if not t_reg:
            return False

        t_sub = t_ext.subdomain.lower() if t_ext.subdomain else ""
        t_host = f"{t_sub}.{t_reg}" if t_sub else t_reg

        for off in official_domains:
            off_clean = off.strip().lower()
            if not off_clean:
                continue
            o_ext = tldextract.extract(off_clean)
            o_reg = (
                o_ext.top_domain_under_public_suffix
                if hasattr(o_ext, "top_domain_under_public_suffix")
                else (o_ext.registered_domain or "")
            ).lower()

            if not o_reg or t_reg != o_reg:
                continue

            o_sub = o_ext.subdomain.lower() if o_ext.subdomain else ""
            o_host = f"{o_sub}.{o_reg}" if o_sub else o_reg

            if o_sub:
                if t_host == o_host or t_host.endswith(f".{o_host}"):
                    return True
            else:
                if t_host == o_reg or t_host.endswith(f".{o_reg}"):
                    return True

        return False
    except Exception:
        return False


class BrandVisualMatcher:
    """Matches live viewport screenshots against indexed reference brands in the Brand Vault

    using combined MobileNetV3 deep embeddings and perceptual hashing (pHash).
    """

    def __init__(
        self,
        index_path: Optional[Union[str, Path]] = None,
        meta_path: Optional[Union[str, Path]] = None,
        embedder: Optional[MobileNetV3Embedder] = None,
    ):
        base_dir = Path(__file__).resolve().parents[2]
        self.index_path = Path(index_path) if index_path else base_dir / "models" / "brand_vault_index.npz"
        self.meta_path = Path(meta_path) if meta_path else base_dir / "models" / "brand_vault_meta.json"
        self.embedder = embedder

        self.brand_meta: dict[str, BrandRecord] = {}
        self.reference_embeddings: np.ndarray = np.empty((0, 576), dtype=np.float32)
        self.reference_brand_ids: list[str] = []
        self.reference_phashes: list[imagehash.ImageHash] = []

        self.load_vault()

    def _get_embedder(self) -> MobileNetV3Embedder:
        if self.embedder is None:
            self.embedder = MobileNetV3Embedder()
        return self.embedder

    def load_vault(self) -> None:
        """Load precomputed Brand Vault embeddings and metadata."""
        if not self.index_path.exists() or not self.meta_path.exists():
            logger.info("Brand Vault index or metadata not found at %s / %s", self.index_path, self.meta_path)
            return

        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta_raw = json.load(f)
                self.brand_meta = {
                    item["brand_id"]: BrandRecord(**item) for item in meta_raw
                }

            npz_data = np.load(self.index_path, allow_pickle=True)
            self.reference_embeddings = npz_data["embeddings"].astype(np.float32)
            self.reference_brand_ids = [str(b) for b in npz_data["brand_ids"]]

            # Load perceptual hashes (stored as hex strings)
            raw_hashes = npz_data["phashes"]
            self.reference_phashes = [
                imagehash.hex_to_hash(str(h)) for h in raw_hashes
            ]
            logger.info(
                "Loaded Brand Vault with %d reference images across %d brands",
                len(self.reference_brand_ids),
                len(self.brand_meta),
            )
        except Exception as e:
            logger.error("Failed to load Brand Vault index: %s", e)

    @staticmethod
    def is_blank_or_uniform(image: Image.Image, std_threshold: float = 5.0) -> bool:
        """Determines whether an image is solid-color, blank, or uniform."""
        try:
            gray = np.array(image.convert("L"), dtype=np.float32)
            return float(np.std(gray)) < std_threshold
        except Exception:
            return True

    def match(
        self,
        target_url: str,
        image: Optional[Image.Image] = None,
        image_bytes: Optional[bytes] = None,
    ) -> VisualCvResult:
        """Matches a target webpage screenshot against the Brand Vault."""
        start_time = time.perf_counter()

        if image is None and not image_bytes:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return VisualCvResult(
                inference_time_ms=elapsed_ms,
                skipped_reason="no_screenshot",
            )

        if image is None and image_bytes:
            try:
                image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                return VisualCvResult(
                    inference_time_ms=elapsed_ms,
                    skipped_reason=f"corrupt_screenshot: {str(e)}",
                )

        # Blank or uniform check
        if self.is_blank_or_uniform(image):
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return VisualCvResult(
                inference_time_ms=elapsed_ms,
                skipped_reason="blank_or_uniform_screenshot",
            )

        # Check if brand vault is empty
        if len(self.reference_brand_ids) == 0 or self.reference_embeddings.shape[0] == 0:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return VisualCvResult(
                inference_time_ms=elapsed_ms,
                skipped_reason="empty_brand_vault",
            )

        # 1. Compute query deep embedding via MobileNetV3
        embedder = self._get_embedder()
        query_vec = embedder.extract_from_image(image)

        # 2. Compute query perceptual hash
        query_phash = imagehash.phash(image)

        # 3. Vectorized Cosine Similarities (both embeddings are L2-normalized)
        cosine_sims = np.dot(self.reference_embeddings, query_vec)
        cosine_sims = np.clip(cosine_sims, 0.0, 1.0)

        # 4. Per-reference comparison and aggregation by brand
        brand_best_candidates: dict[str, VisualMatchCandidate] = {}

        for idx, (b_id, cos_sim, ref_hash) in enumerate(
            zip(self.reference_brand_ids, cosine_sims, self.reference_phashes)
        ):
            phash_dist = int(query_phash - ref_hash)
            phash_dist = max(0, min(64, phash_dist))
            phash_sim = max(0.0, 1.0 - (phash_dist / 64.0))

            # Weighted fusion formula: 0.75 * cosine + 0.25 * phash_similarity
            combined_score = float(0.75 * float(cos_sim) + 0.25 * phash_sim)
            combined_score = max(0.0, min(1.0, combined_score))

            brand_info = self.brand_meta.get(b_id)
            brand_name = brand_info.brand_name if brand_info else b_id

            candidate = VisualMatchCandidate(
                brand_id=b_id,
                brand_name=brand_name,
                cosine_similarity=float(cos_sim),
                phash_distance=phash_dist,
                phash_similarity=phash_sim,
                combined_visual_score=combined_score,
            )

            # Keep candidate with highest combined_visual_score per brand
            if (
                b_id not in brand_best_candidates
                or candidate.combined_visual_score
                > brand_best_candidates[b_id].combined_visual_score
            ):
                brand_best_candidates[b_id] = candidate

        # Sort candidate brands descending by combined_visual_score
        sorted_candidates = sorted(
            brand_best_candidates.values(),
            key=lambda c: c.combined_visual_score,
            reverse=True,
        )

        top_candidates = sorted_candidates[:3]
        top_match = sorted_candidates[0] if sorted_candidates else None

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        if not top_match:
            return VisualCvResult(
                inference_time_ms=elapsed_ms,
                skipped_reason="no_candidates",
            )

        matched_brand_record = self.brand_meta.get(top_match.brand_id)
        official_domains = (
            matched_brand_record.official_domains if matched_brand_record else []
        )

        domain_is_official = is_official_domain(target_url, official_domains)

        # Multi-Modal Brand Impersonation Trigger Rule:
        # If visual_similarity >= 0.82 for a protected brand AND registered_domain
        # is NOT in that brand's official_domains whitelist, OR (cosine >= 0.75 AND phash_distance <= 10)
        impersonation_detected = (not domain_is_official) and (
            top_match.cosine_similarity >= 0.82
            or (top_match.cosine_similarity >= 0.75 and top_match.phash_distance <= 10)
        )

        return VisualCvResult(
            matched_brand_id=top_match.brand_id,
            matched_brand_name=top_match.brand_name,
            official_domains=official_domains,
            cosine_similarity=top_match.cosine_similarity,
            phash_distance=top_match.phash_distance,
            combined_visual_score=top_match.combined_visual_score,
            domain_is_official=domain_is_official,
            impersonation_detected=impersonation_detected,
            top_candidates=top_candidates,
            inference_time_ms=elapsed_ms,
            skipped_reason=None,
        )
