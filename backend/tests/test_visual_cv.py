from __future__ import annotations

import io
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from backend.app.core.visual_cv import (
    MobileNetV3Embedder,
    BrandVisualMatcher,
    is_official_domain,
)
from backend.app.schemas import BrandRecord
from backend.build_brand_vault import draw_synthetic_screenshot


@pytest.fixture(scope="module")
def sample_sbi_image():
    """Deterministic synthetic SBI login screenshot."""
    return draw_synthetic_screenshot("sbi", "State Bank of India")


@pytest.fixture(scope="module")
def sample_hdfc_image():
    """Deterministic synthetic HDFC login screenshot."""
    return draw_synthetic_screenshot("hdfc", "HDFC Bank")


class TestDomainOwnership:
    """Test suite for is_official_domain helper."""

    def test_exact_domain_match(self):
        officials = ["onlinesbi.sbi", "sbi.co.in", "onlinesbi.com"]
        assert is_official_domain("https://onlinesbi.sbi", officials) is True
        assert is_official_domain("https://sbi.co.in/portal", officials) is True

    def test_valid_subdomain_match(self):
        officials = ["onlinesbi.sbi", "google.com"]
        assert is_official_domain("https://retail.onlinesbi.sbi/retail/login.htm", officials) is True
        assert is_official_domain("https://accounts.google.com/signin", officials) is True
        assert is_official_domain("https://deep.sub.retail.onlinesbi.sbi", officials) is True

    def test_subdomain_spoofing_rejected(self):
        """Attacker uses brand name inside subdomain of attacker domain."""
        officials = ["onlinesbi.sbi", "sbi.co.in"]
        assert is_official_domain("https://onlinesbi.sbi.evil.xyz/login", officials) is False
        assert is_official_domain("https://onlinesbi.sbi.attacker.com", officials) is False

    def test_hyphenated_lookalike_domain_rejected(self):
        """Attacker registers typosquatting / hyphenated domain."""
        officials = ["onlinesbi.sbi"]
        assert is_official_domain("https://fake-onlinesbi.sbi", officials) is False
        assert is_official_domain("https://onlinesbi-sbi.com", officials) is False

    def test_path_attack_rejected(self):
        officials = ["google.com"]
        assert is_official_domain("https://evil.com/accounts.google.com", officials) is False

    def test_campus_multi_level_domain(self):
        officials = ["mitaoe.ac.in"]
        assert is_official_domain("https://mitaoe.ac.in", officials) is True
        assert is_official_domain("https://portal.mitaoe.ac.in", officials) is True
        assert is_official_domain("https://mitaoe.ac.in.fake.org", officials) is False

    def test_empty_and_invalid_inputs(self):
        assert is_official_domain("", ["google.com"]) is False
        assert is_official_domain("not-a-url", ["google.com"]) is False
        assert is_official_domain("https://google.com", []) is False


class TestBlankScreenshotDetection:
    """Test suite for blank or uniform page detection."""

    def test_completely_white_screenshot(self):
        img = Image.new("RGB", (1280, 720), (255, 255, 255))
        assert BrandVisualMatcher.is_blank_or_uniform(img) is True

    def test_completely_black_screenshot(self):
        img = Image.new("RGB", (1280, 720), (0, 0, 0))
        assert BrandVisualMatcher.is_blank_or_uniform(img) is True

    def test_near_uniform_screenshot(self):
        # Very low variance noise
        arr = np.full((720, 1280, 3), 240, dtype=np.uint8)
        arr[0, 0, :] = 241
        img = Image.fromarray(arr)
        assert BrandVisualMatcher.is_blank_or_uniform(img, std_threshold=5.0) is True

    def test_normal_rich_screenshot(self, sample_sbi_image):
        assert BrandVisualMatcher.is_blank_or_uniform(sample_sbi_image) is False


class TestBrandMatcherEdgeCases:
    """Test matcher behavior on missing or corrupted screenshots."""

    def test_missing_screenshot(self):
        matcher = BrandVisualMatcher()
        result = matcher.match("https://evil.com")
        assert result.skipped_reason == "no_screenshot"
        assert result.impersonation_detected is False

    def test_corrupt_screenshot_bytes(self):
        matcher = BrandVisualMatcher()
        result = matcher.match("https://evil.com", image_bytes=b"not an image file")
        assert result.skipped_reason is not None
        assert "corrupt_screenshot" in result.skipped_reason
        assert result.impersonation_detected is False

    def test_blank_screenshot_handling(self):
        matcher = BrandVisualMatcher()
        blank_img = Image.new("RGB", (1280, 720), (250, 250, 250))
        result = matcher.match("https://evil.com", image=blank_img)
        assert result.skipped_reason == "blank_or_uniform_screenshot"
        assert result.impersonation_detected is False

    def test_empty_vault_handling(self, sample_sbi_image, tmp_path):
        fake_index = tmp_path / "nonexistent.npz"
        fake_meta = tmp_path / "nonexistent.json"
        matcher = BrandVisualMatcher(index_path=fake_index, meta_path=fake_meta)
        result = matcher.match("https://sbi.co.in", image=sample_sbi_image)
        assert result.skipped_reason == "empty_brand_vault"
        assert result.impersonation_detected is False


class DummyEmbedder:
    """Deterministic mock embedder for testing visual matcher math."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def extract_from_image(self, image: Image.Image) -> np.ndarray:
        # Generate deterministic vector based on image average color
        avg_color = np.array(image.resize((1, 1))).flatten().astype(np.float32)
        base = np.zeros(576, dtype=np.float32)
        base[:len(avg_color)] = avg_color
        norm = np.linalg.norm(base)
        return (base / norm).astype(np.float32) if norm > 0 else base


class TestBrandVaultMatching:
    """Test suite for BrandVisualMatcher fusion logic and impersonation rules."""

    @pytest.fixture
    def mock_brand_vault(self, tmp_path, sample_sbi_image, sample_hdfc_image):
        """Construct a minimal Brand Vault index for SBI and HDFC."""
        import imagehash

        embedder = DummyEmbedder()
        sbi_vec = embedder.extract_from_image(sample_sbi_image)
        hdfc_vec = embedder.extract_from_image(sample_hdfc_image)

        sbi_hash = str(imagehash.phash(sample_sbi_image))
        hdfc_hash = str(imagehash.phash(sample_hdfc_image))

        embeddings = np.vstack([sbi_vec, hdfc_vec])
        brand_ids = np.array(["sbi", "hdfc"], dtype=object)
        phashes = np.array([sbi_hash, hdfc_hash], dtype=object)

        index_file = tmp_path / "test_brand_vault_index.npz"
        np.savez_compressed(index_file, embeddings=embeddings, brand_ids=brand_ids, phashes=phashes)

        meta_file = tmp_path / "test_brand_vault_meta.json"
        meta_data = [
            {
                "brand_id": "sbi",
                "brand_name": "State Bank of India",
                "official_domains": ["onlinesbi.sbi", "sbi.co.in"],
                "category": "banking",
                "reference_images": ["sbi_login.png"],
            },
            {
                "brand_id": "hdfc",
                "brand_name": "HDFC Bank",
                "official_domains": ["hdfcbank.com"],
                "category": "banking",
                "reference_images": ["hdfc_netbanking.png"],
            },
        ]
        import json
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_data, f)

        matcher = BrandVisualMatcher(
            index_path=index_file,
            meta_path=meta_file,
            embedder=embedder,
        )
        return matcher

    def test_official_domain_match_no_impersonation(self, mock_brand_vault, sample_sbi_image):
        """When visual matches SBI and target is on retail.onlinesbi.sbi, no impersonation."""
        result = mock_brand_vault.match("https://retail.onlinesbi.sbi/retail/login.htm", image=sample_sbi_image)
        assert result.matched_brand_id == "sbi"
        assert result.matched_brand_name == "State Bank of India"
        assert result.domain_is_official is True
        assert result.impersonation_detected is False
        assert result.cosine_similarity > 0.95
        assert result.phash_distance == 0
        assert result.combined_visual_score > 0.95
        assert len(result.top_candidates) == 2

    def test_phishing_domain_triggers_impersonation(self, mock_brand_vault, sample_sbi_image):
        """When visual matches SBI but domain is unauthorized, impersonation detected."""
        result = mock_brand_vault.match("https://onlinesbi.sbi.evil-phishing.xyz", image=sample_sbi_image)
        assert result.matched_brand_id == "sbi"
        assert result.domain_is_official is False
        assert result.impersonation_detected is True
        assert result.cosine_similarity >= 0.82

    def test_typosquatting_triggers_impersonation(self, mock_brand_vault, sample_sbi_image):
        """When visual matches SBI but domain is lookalike fake-onlinesbi.sbi, impersonation detected."""
        result = mock_brand_vault.match("https://fake-onlinesbi.sbi", image=sample_sbi_image)
        assert result.matched_brand_id == "sbi"
        assert result.domain_is_official is False
        assert result.impersonation_detected is True

    def test_candidate_ranking_and_score_formula(self, mock_brand_vault, sample_hdfc_image):
        """Verify candidates are sorted descending by combined_visual_score."""
        result = mock_brand_vault.match("https://netbanking.hdfcbank.com", image=sample_hdfc_image)
        assert result.matched_brand_id == "hdfc"
        assert result.domain_is_official is True
        assert result.impersonation_detected is False

        # Verify combined visual score formula
        top_cand = result.top_candidates[0]
        expected_combined = 0.75 * top_cand.cosine_similarity + 0.25 * top_cand.phash_similarity
        assert top_cand.combined_visual_score == pytest.approx(expected_combined, abs=1e-5)

        # Check descending sort
        scores = [c.combined_visual_score for c in result.top_candidates]
        assert scores == sorted(scores, reverse=True)

    def test_default_production_brand_vault(self, sample_sbi_image):
        """Verify default BrandVisualMatcher loads production brand vault and detects SBI."""
        matcher = BrandVisualMatcher()
        assert len(matcher.brand_meta) >= 15
        assert len(matcher.reference_brand_ids) >= 15
        assert matcher.reference_embeddings.shape[1] == 576

        # Test SBI match on official domain
        res_off = matcher.match("https://onlinesbi.sbi", image=sample_sbi_image)
        assert res_off.matched_brand_id == "sbi"
        assert res_off.domain_is_official is True
        assert res_off.impersonation_detected is False

        # Test SBI match on phishing domain
        res_phish = matcher.match("https://sbi-login.fraudulent-site.xyz", image=sample_sbi_image)
        assert res_phish.matched_brand_id == "sbi"
        assert res_phish.domain_is_official is False
        assert res_phish.impersonation_detected is True


class TestMobileNetV3Integration:
    """Test suite for live MobileNetV3 feature extractor."""

    def test_mobilenet_feature_shape_and_norm(self, sample_sbi_image):
        embedder = MobileNetV3Embedder(device="cpu")
        vec = embedder.extract_from_image(sample_sbi_image)
        assert vec.shape == (576,)
        assert vec.dtype == np.float32
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    def test_mobilenet_extract_from_bytes(self, sample_sbi_image):
        embedder = MobileNetV3Embedder(device="cpu")
        buf = io.BytesIO()
        sample_sbi_image.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        vec_from_bytes = embedder.extract_from_bytes(raw_bytes)
        vec_from_image = embedder.extract_from_image(sample_sbi_image)
        assert np.allclose(vec_from_bytes, vec_from_image, atol=1e-5)



