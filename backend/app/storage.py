from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import Optional, Union

from .schemas import ScanHistoryItem

logger = logging.getLogger("brandshield.storage")


class ScanHistoryStore:
    """Thread-safe, non-blocking scan history store backed by standard library SQLite."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            base_dir = Path(__file__).resolve().parents[2]
            self.db_path = base_dir / "data" / "scan_history.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create the scan_history table and indexes if they do not exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS scan_history (
                        request_id TEXT PRIMARY KEY,
                        analyzed_at TEXT NOT NULL,
                        original_url TEXT NOT NULL,
                        final_url TEXT NOT NULL,
                        verdict TEXT NOT NULL,
                        risk_score_pct INTEGER NOT NULL,
                        matched_brand_name TEXT,
                        impersonation_detected INTEGER NOT NULL,
                        processing_time_ms INTEGER NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_scan_history_analyzed_at
                    ON scan_history(analyzed_at DESC)
                    """
                )
                conn.commit()
            logger.info("Initialized ScanHistoryStore database at %s", self.db_path)
        except Exception as e:
            logger.error("Failed to initialize SQLite database at %s: %s", self.db_path, e)

    def _sync_log_scan(self, item: ScanHistoryItem) -> None:
        """Synchronous insert worker for log_scan."""
        with sqlite3.connect(self.db_path, timeout=5.0) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scan_history (
                    request_id, analyzed_at, original_url, final_url,
                    verdict, risk_score_pct, matched_brand_name,
                    impersonation_detected, processing_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.request_id,
                    item.analyzed_at,
                    item.original_url,
                    item.final_url,
                    item.verdict,
                    item.risk_score_pct,
                    item.matched_brand_name,
                    1 if item.impersonation_detected else 0,
                    item.processing_time_ms,
                ),
            )
            conn.commit()

    async def log_scan(self, item: ScanHistoryItem) -> None:
        """Asynchronously log an analyzed scan to the SQLite audit log."""
        try:
            await asyncio.to_thread(self._sync_log_scan, item)
        except Exception as e:
            logger.error("Error logging scan to history DB (%s): %s", item.request_id, e)

    def _sync_get_recent(self, limit: int) -> list[ScanHistoryItem]:
        """Synchronous query worker for get_recent."""
        items: list[ScanHistoryItem] = []
        with sqlite3.connect(self.db_path, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT request_id, analyzed_at, original_url, final_url,
                       verdict, risk_score_pct, matched_brand_name,
                       impersonation_detected, processing_time_ms
                FROM scan_history
                ORDER BY analyzed_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            for row in cursor.fetchall():
                items.append(
                    ScanHistoryItem(
                        request_id=row["request_id"],
                        analyzed_at=row["analyzed_at"],
                        original_url=row["original_url"],
                        final_url=row["final_url"],
                        verdict=row["verdict"],
                        risk_score_pct=row["risk_score_pct"],
                        matched_brand_name=row["matched_brand_name"],
                        impersonation_detected=bool(row["impersonation_detected"]),
                        processing_time_ms=row["processing_time_ms"],
                    )
                )
        return items

    async def get_recent(self, limit: int = 20) -> list[ScanHistoryItem]:
        """Asynchronously fetch the most recent scans ordered descending by timestamp."""
        try:
            return await asyncio.to_thread(self._sync_get_recent, limit)
        except Exception as e:
            logger.error("Error fetching recent scan history: %s", e)
            return []
