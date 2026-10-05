from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from app.models import HistorySummary, TrendItem


class HistoryService:
    def __init__(self, history_file: Path) -> None:
        self.history_file = history_file
        if not self.history_file.exists():
            self.history_file.write_text("[]", encoding="utf-8")

    def append(self, record: dict[str, Any]) -> None:
        items = self._load()
        items.append(record)
        self.history_file.write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def summary(self, student_name: Optional[str] = None) -> HistorySummary:
        items = self._load()
        if student_name:
            items = [item for item in items if item.get("student_name") == student_name]
        counter = Counter(item.get("prerequisite_gap") for item in items if item.get("prerequisite_gap"))
        top_blind_spots = [
            TrendItem(knowledge_point=name, count=count)
            for name, count in counter.most_common(5)
        ]
        return HistorySummary(total_records=len(items), top_blind_spots=top_blind_spots)

    def build_diagnosis_context(
        self,
        *,
        student_name: Optional[str],
        focus_knowledge_points: list[str],
    ) -> dict[str, Any]:
        items = self._load()
        if student_name:
            items = [item for item in items if item.get("student_name") == student_name]

        normalized_focus = [item for item in dict.fromkeys(focus_knowledge_points) if item]
        related_items = []
        for item in items:
            record_points = self._normalize_text_list(item.get("knowledge_points"))
            record_gap = str(item.get("prerequisite_gap") or "").strip()
            if any(point in record_points for point in normalized_focus) or (record_gap and record_gap in normalized_focus):
                related_items.append(item)

        gap_counter = Counter(
            item.get("prerequisite_gap")
            for item in related_items
            if item.get("prerequisite_gap")
        )
        level_counter = Counter(
            item.get("error_level")
            for item in related_items
            if item.get("error_level")
        )
        recent_examples = []
        for item in related_items[-5:]:
            recent_examples.append(
                {
                    "prerequisite_gap": item.get("prerequisite_gap", ""),
                    "error_level": item.get("error_level", ""),
                    "knowledge_points": self._normalize_text_list(item.get("knowledge_points")),
                }
            )

        return {
            "student_record_count": len(items),
            "related_record_count": len(related_items),
            "focus_knowledge_points": normalized_focus,
            "top_repeated_gaps": [
                {"knowledge_point": name, "count": count}
                for name, count in gap_counter.most_common(5)
            ],
            "error_level_distribution": dict(level_counter),
            "recent_examples": recent_examples,
        }

    def _load(self) -> list[dict[str, Any]]:
        return json.loads(self.history_file.read_text(encoding="utf-8"))

    @staticmethod
    def _normalize_text_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        return []
