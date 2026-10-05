from __future__ import annotations

from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any


class KnowledgeGraphService:
    GRAPH_RELATIONS = {"prerequisites_for", "relates_to", "is_a"}
    MANUAL_ALIAS_HINTS = {
        "分数通分应用题": "分数通分",
        "分数加法应用题": "分数加法",
        "分数意义问题": "分数意义",
        "分数意义相关题": "分数意义",
        "最小公倍数问题": "最小公倍数",
        "磁感线方向判断": "磁场方向",
        "磁场方向判断": "磁场方向",
    }
    GENERIC_SUFFIXES = (
        "相关知识点",
        "知识点",
        "应用题",
        "计算题",
        "判断题",
        "题目",
        "题型",
        "专题",
        "练习",
        "训练",
        "问题",
        "错误",
        "方法",
        "概念",
        "能力",
        "考点",
        "相关题",
        "相关",
        "试题",
        "习题",
        "题",
    )
    GENERIC_PREFIXES = ("关于", "有关", "针对", "学生在", "学生对", "围绕", "基于")

    def __init__(self, graph_path: Path) -> None:
        self.graph_path = graph_path
        self._reload()

    def _reload(self) -> None:
        payload = self._load_payload(self.graph_path)
        self.nodes: dict[str, dict[str, Any]] = {}
        self.node_names_by_id: dict[str, str] = {}
        self.alias_to_name: dict[str, str] = {}
        self.normalized_alias_to_names: dict[str, list[str]] = {}
        self.alias_entries: list[tuple[str, str, str]] = []
        self.edges: list[dict[str, str]] = []
        self.edge_keys: set[tuple[str, str, str]] = set()
        self.outgoing_edges: dict[str, list[dict[str, str]]] = {}
        self.incoming_edges: dict[str, list[dict[str, str]]] = {}
        self._load_nodes(payload)
        self._load_edges(payload)
        self._load_embedded_prerequisites()
        self._register_manual_aliases()

    def reload(self) -> None:
        self._reload()

    def get_prerequisites(self, node_name: str) -> list[str]:
        node = self.nodes.get(self.normalize_node_name(node_name), {})
        return list(node.get("prerequisites", []))

    def get_all_paths(self, node_name: str) -> list[list[str]]:
        canonical_name = self.normalize_node_name(node_name)
        if canonical_name not in self.nodes:
            return []
        paths: list[list[str]] = []
        self._walk(canonical_name, [], paths)
        return paths or [[node_name]]

    def get_lowest_prerequisite(self, node_name: str) -> str:
        canonical_name = self.normalize_node_name(node_name)
        paths = self.get_all_paths(node_name)
        if not paths:
            return canonical_name or node_name
        longest_path = max(paths, key=len)
        return longest_path[0]

    def build_learning_snapshot(self, *, focus_name: str, prerequisite_gap: str = "") -> dict[str, Any]:
        focus_name = self.normalize_node_name(focus_name)
        prerequisite_gap = self.normalize_node_name(prerequisite_gap)
        if focus_name not in self.nodes:
            return {
                "focus_node": focus_name,
                "prerequisite_gap": prerequisite_gap,
                "node_count": 0,
                "edge_count": 0,
                "nodes": [],
                "edges": [],
            }

        graph_paths = self.get_all_paths(focus_name)
        path_nodes = self._unique_path_nodes(graph_paths)
        gap_name = prerequisite_gap or self.get_lowest_prerequisite(focus_name)

        layers = self._build_layers_from_paths(graph_paths)
        selected_nodes = set(path_nodes)
        related_map: dict[str, str] = {}

        focus_successors = self._neighbor_names(focus_name, relation="prerequisites_for", direction="outgoing", limit=4)
        for name in focus_successors:
            if name not in selected_nodes:
                selected_nodes.add(name)
                layers.setdefault(name, 1)
                related_map[name] = "next"

        anchor_nodes = [focus_name]
        if gap_name and gap_name in self.nodes and gap_name != focus_name:
            anchor_nodes.append(gap_name)
        for anchor in anchor_nodes:
            for relation in ("relates_to", "is_a"):
                relation_neighbors = self._neighbor_names(anchor, relation=relation, direction="both", limit=3)
                for name in relation_neighbors:
                    if name not in selected_nodes:
                        selected_nodes.add(name)
                        layers.setdefault(name, layers.get(anchor, 0))
                        related_map[name] = "related"

        nodes = []
        for name in sorted(selected_nodes, key=lambda item: (layers.get(item, 0), item)):
            node = self.nodes.get(name, {})
            role = "path"
            if name == focus_name:
                role = "focus"
            elif gap_name and name == gap_name:
                role = "gap"
            elif related_map.get(name):
                role = related_map[name]
            nodes.append(
                {
                    "name": name,
                    "subject": node.get("subject", ""),
                    "node_type": node.get("node_type", ""),
                    "role": role,
                    "layer": layers.get(name, 0),
                    "description": node.get("description", ""),
                }
            )

        edges = []
        for edge in self.edges:
            if edge["relation"] not in self.GRAPH_RELATIONS:
                continue
            if edge["source"] not in selected_nodes or edge["target"] not in selected_nodes:
                continue
            edges.append(
                {
                    "source": edge["source"],
                    "target": edge["target"],
                    "relation": edge["relation"],
                    "evidence": edge.get("evidence", ""),
                }
            )

        return {
            "focus_node": focus_name,
            "prerequisite_gap": gap_name,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
        }

    def has_node(self, node_name: str) -> bool:
        return self.normalize_node_name(node_name) in self.nodes

    def find_missing_knowledge_points(self, knowledge_points: list[str]) -> list[str]:
        missing: list[str] = []
        for item in knowledge_points:
            text = self._first_text(item)
            if not text:
                continue
            if self.has_node(text):
                continue
            if text not in missing:
                missing.append(text)
        return missing

    def search_similar_nodes(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        raw_query = self._first_text(query)
        if not raw_query:
            return []
        scored: list[tuple[float, str]] = []
        seen: set[str] = set()
        for node_name in self.nodes:
            score = self._score_candidate(raw_query, node_name, self._normalize_lookup_text(node_name))
            if score <= 0.0 or node_name in seen:
                continue
            seen.add(node_name)
            scored.append((score, node_name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        results = []
        for score, node_name in scored[:limit]:
            node = self.nodes.get(node_name, {})
            results.append(
                {
                    "name": node_name,
                    "subject": node.get("subject", ""),
                    "node_type": node.get("node_type", ""),
                    "description": node.get("description", ""),
                    "score": round(score, 4),
                }
            )
        return results

    def add_graph_candidates(self, candidates: list[dict[str, Any]]) -> list[str]:
        payload = self._load_mutable_payload()
        nodes = payload.setdefault("nodes", [])
        edges = payload.setdefault("edges", [])
        added_nodes: list[str] = []
        for candidate in candidates:
            node_name = self._first_text(candidate.get("name"))
            if not node_name:
                continue
            node_name = self.normalize_node_name(node_name) if self.has_node(node_name) else node_name
            canonical_node = self._upsert_node_record(
                nodes=nodes,
                name=node_name,
                subject=self._first_text(candidate.get("subject")),
                node_type=self._first_text(candidate.get("node_type"), "Knowledge"),
                description=self._first_text(candidate.get("description")),
                aliases=self._normalize_alias_values(candidate.get("aliases")),
            )
            if canonical_node not in added_nodes:
                added_nodes.append(canonical_node)
            prerequisites = self._normalize_relation_candidates(candidate.get("prerequisites"))
            related_to = self._normalize_relation_candidates(candidate.get("related_to"))
            is_a = self._normalize_relation_candidates(candidate.get("is_a"))
            for prerequisite in prerequisites:
                source_name = self._ensure_mutable_node(nodes, prerequisite)
                self._append_edge_record(edges, source_name, canonical_node, "prerequisites_for")
            for related_name in related_to:
                target_name = self._ensure_mutable_node(nodes, related_name)
                self._append_edge_record(edges, canonical_node, target_name, "relates_to")
            for parent_name in is_a:
                target_name = self._ensure_mutable_node(nodes, parent_name)
                self._append_edge_record(edges, canonical_node, target_name, "is_a")
        self._write_payload(payload)
        self.reload()
        return added_nodes

    def _walk(self, node_name: str, trail: list[str], paths: list[list[str]]) -> None:
        trail = trail + [node_name]
        prerequisites = self.get_prerequisites(node_name)
        if not prerequisites:
            paths.append(list(reversed(trail)))
            return
        for prerequisite in prerequisites:
            self._walk(prerequisite, trail, paths)

    def _load_mutable_payload(self) -> dict[str, Any]:
        payload = self._load_payload(self.graph_path)
        if isinstance(payload, dict):
            payload.setdefault("nodes", [])
            payload.setdefault("edges", [])
            return payload
        return {"nodes": list(payload) if isinstance(payload, list) else [], "edges": []}

    def _write_payload(self, payload: dict[str, Any]) -> None:
        self.graph_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _find_node_record(self, nodes: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
        for node in nodes:
            if self._extract_node_name(node) == name:
                return node
        return None

    def _upsert_node_record(
        self,
        *,
        nodes: list[dict[str, Any]],
        name: str,
        subject: str,
        node_type: str,
        description: str,
        aliases: list[str],
    ) -> str:
        existing_name = self.normalize_node_name(name)
        canonical_name = existing_name if existing_name in self.nodes else name
        node = self._find_node_record(nodes, canonical_name)
        if node is None:
            node = {
                "name": canonical_name,
                "subject": subject,
                "type": node_type or "Knowledge",
            }
            if description or aliases:
                node["properties"] = {}
            nodes.append(node)
        if subject and not node.get("subject"):
            node["subject"] = subject
        if node_type and not node.get("type"):
            node["type"] = node_type
        properties = node.get("properties")
        if not isinstance(properties, dict):
            properties = {}
            if description or aliases:
                node["properties"] = properties
        if description:
            properties["description"] = description
        if aliases:
            merged_aliases = self._normalize_alias_values(properties.get("aliases")) + aliases
            unique_aliases: list[str] = []
            for alias in merged_aliases:
                if alias != canonical_name and alias not in unique_aliases:
                    unique_aliases.append(alias)
            if unique_aliases:
                properties["aliases"] = unique_aliases
        return canonical_name

    def _normalize_relation_candidates(self, value: Any) -> list[str]:
        names: list[str] = []
        for item in self._normalize_alias_values(value):
            canonical = self.normalize_node_name(item)
            name = canonical if canonical in self.nodes else item
            if name not in names:
                names.append(name)
        return names

    def _ensure_mutable_node(self, nodes: list[dict[str, Any]], name: str) -> str:
        canonical_name = self.normalize_node_name(name)
        final_name = canonical_name if canonical_name in self.nodes else name
        if self._find_node_record(nodes, final_name) is None:
            nodes.append({"name": final_name, "type": "Knowledge"})
        return final_name

    def _append_edge_record(self, edges: list[dict[str, Any]], source_name: str, target_name: str, relation: str) -> None:
        for edge in edges:
            if (
                self._resolve_endpoint_name(edge.get("source")) == source_name
                and self._resolve_endpoint_name(edge.get("target")) == target_name
                and self._first_text(edge.get("relation"), edge.get("type"), edge.get("edge_type"), edge.get("predicate")) == relation
            ):
                return
        edges.append({"source": source_name, "target": target_name, "relation": relation})

    def _load_payload(self, graph_path: Path) -> Any:
        raw_text = graph_path.read_text(encoding="utf-8").strip()
        if graph_path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in raw_text.splitlines() if line.strip()]
        return json.loads(raw_text)

    def _load_nodes(self, payload: Any) -> None:
        for item in self._iter_node_records(payload):
            name = self._extract_node_name(item)
            if not name:
                continue
            properties = item.get("properties") if isinstance(item.get("properties"), dict) else {}
            subject = self._first_text(
                item.get("subject"),
                item.get("domain"),
                item.get("discipline"),
                properties.get("subject"),
            )
            node = self.nodes.setdefault(
                name,
                {
                    "name": name,
                    "subject": subject,
                    "node_type": self._first_text(item.get("label"), item.get("type"), "Knowledge"),
                    "description": self._first_text(
                        properties.get("definition"),
                        properties.get("description"),
                        properties.get("relations"),
                    ),
                    "prerequisites": [],
                },
            )
            if subject and not node.get("subject"):
                node["subject"] = subject
            node_type = self._first_text(item.get("label"), item.get("type"))
            if node_type and node.get("node_type") in {"", "Knowledge"}:
                node["node_type"] = node_type
            description = self._first_text(
                properties.get("definition"),
                properties.get("description"),
                properties.get("relations"),
            )
            if description and not node.get("description"):
                node["description"] = description
            for prerequisite in self._normalize_name_list(item.get("prerequisites")):
                if prerequisite not in node["prerequisites"]:
                    node["prerequisites"].append(prerequisite)
            node_id = self._first_text(item.get("id"), item.get("node_id"), item.get("entity_id"))
            if node_id:
                self.node_names_by_id[node_id] = name
            self._register_aliases(name, item, properties)

    def _load_edges(self, payload: Any) -> None:
        for edge in self._iter_edge_records(payload):
            relation = self._first_text(
                edge.get("relation"),
                edge.get("type"),
                edge.get("edge_type"),
                edge.get("predicate"),
            )
            source_name = self._resolve_endpoint_name(
                edge.get("source", edge.get("src", edge.get("from", edge.get("head"))))
            ) or self._first_text(edge.get("source_name"))
            target_name = self._resolve_endpoint_name(
                edge.get("target", edge.get("dst", edge.get("to", edge.get("tail"))))
            ) or self._first_text(edge.get("target_name"))
            evidence = ""
            properties = edge.get("properties") if isinstance(edge.get("properties"), dict) else {}
            if properties:
                evidence = self._first_text(properties.get("evidence"), properties.get("relations"))
            self._append_edge(source_name, target_name, relation, evidence=evidence)

    def _load_embedded_prerequisites(self) -> None:
        for node in self.nodes.values():
            target_name = node["name"]
            for prerequisite in node.get("prerequisites", []):
                self._append_edge(prerequisite, target_name, "prerequisites_for")

    def _append_edge(self, source_name: str, target_name: str, relation: str, evidence: str = "") -> None:
        if not source_name or not target_name or not relation:
            return
        self._ensure_node(source_name)
        self._ensure_node(target_name)
        edge_key = (source_name, target_name, relation)
        if edge_key in self.edge_keys:
            return
        edge = {
            "source": source_name,
            "target": target_name,
            "relation": relation,
            "evidence": evidence,
        }
        self.edge_keys.add(edge_key)
        self.edges.append(edge)
        self.outgoing_edges.setdefault(source_name, []).append(edge)
        self.incoming_edges.setdefault(target_name, []).append(edge)
        if relation == "prerequisites_for":
            prerequisites = self.nodes[target_name].setdefault("prerequisites", [])
            if source_name not in prerequisites:
                prerequisites.append(source_name)

    def _ensure_node(self, node_name: str) -> None:
        self.nodes.setdefault(
            node_name,
            {
                "name": node_name,
                "subject": "",
                "node_type": "Knowledge",
                "description": "",
                "prerequisites": [],
            },
        )
        self._register_alias(node_name, node_name)

    def _neighbor_names(self, node_name: str, *, relation: str, direction: str, limit: int) -> list[str]:
        names: list[str] = []
        if direction in {"outgoing", "both"}:
            for edge in self.outgoing_edges.get(node_name, []):
                if edge["relation"] == relation:
                    names.append(edge["target"])
        if direction in {"incoming", "both"}:
            for edge in self.incoming_edges.get(node_name, []):
                if edge["relation"] == relation:
                    names.append(edge["source"])
        unique: list[str] = []
        for name in names:
            if name == node_name or name in unique:
                continue
            unique.append(name)
        return unique[:limit]

    @staticmethod
    def _unique_path_nodes(paths: list[list[str]]) -> list[str]:
        ordered: list[str] = []
        for path in paths:
            for node_name in path:
                if node_name not in ordered:
                    ordered.append(node_name)
        return ordered

    @staticmethod
    def _build_layers_from_paths(paths: list[list[str]]) -> dict[str, int]:
        layers: dict[str, int] = {}
        for path in paths:
            focus_index = len(path) - 1
            for index, node_name in enumerate(path):
                layer = index - focus_index
                if node_name not in layers or layer < layers[node_name]:
                    layers[node_name] = layer
        return layers

    def _iter_node_records(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            records = payload.get("nodes")
            if isinstance(records, list):
                return [item for item in records if isinstance(item, dict)]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict) and self._extract_node_name(item)]
        return []

    def _iter_edge_records(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            records = payload.get("edges")
            if isinstance(records, list):
                return [item for item in records if isinstance(item, dict)]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict) and self._looks_like_edge_record(item)]
        return []

    def _resolve_endpoint_name(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._extract_node_name(value) or self.node_names_by_id.get(
                self._first_text(value.get("id"), value.get("node_id"), value.get("entity_id")),
                "",
            )
        if isinstance(value, (int, float)):
            value = str(value)
        if isinstance(value, str):
            return self.node_names_by_id.get(value, value)
        return ""

    def _extract_node_name(self, item: dict[str, Any]) -> str:
        return self._first_text(
            item.get("name"),
            item.get("label"),
            item.get("display_name"),
            item.get("title"),
        )

    def _looks_like_edge_record(self, item: dict[str, Any]) -> bool:
        relation = self._first_text(
            item.get("relation"),
            item.get("type"),
            item.get("edge_type"),
            item.get("predicate"),
        )
        return bool(relation and ("source" in item or "src" in item or "from" in item or "head" in item))

    def _normalize_name_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [self._resolve_endpoint_name(item) for item in value if self._resolve_endpoint_name(item)]
        resolved = self._resolve_endpoint_name(value)
        return [resolved] if resolved else []

    def normalize_node_name(self, node_name: str) -> str:
        raw_name = self._first_text(node_name)
        if not raw_name:
            return ""
        if raw_name in self.nodes:
            return raw_name
        if raw_name in self.alias_to_name:
            return self.alias_to_name[raw_name]

        for variant in self._generate_lookup_variants(raw_name):
            candidates = self.normalized_alias_to_names.get(variant, [])
            if candidates:
                return self._select_best_candidate(raw_name, candidates)

        best_name = ""
        best_score = 0.0
        for canonical_name, alias_text, alias_normalized in self.alias_entries:
            score = self._score_candidate(raw_name, alias_text, alias_normalized)
            if score > best_score:
                best_score = score
                best_name = canonical_name
        if best_score >= 0.66:
            return best_name
        return raw_name

    def _register_aliases(self, node_name: str, item: dict[str, Any], properties: dict[str, Any]) -> None:
        aliases: set[str] = {node_name}
        for container in (item, properties):
            aliases.update(self._normalize_alias_values(container.get("aliases")))
            aliases.update(self._normalize_alias_values(container.get("alias")))
            aliases.update(self._normalize_alias_values(container.get("synonyms")))
            aliases.update(self._normalize_alias_values(container.get("alt_names")))
            aliases.update(self._normalize_alias_values(container.get("keywords")))
        aliases.update(self._generate_lookup_variants(node_name))
        for alias in aliases:
            self._register_alias(node_name, alias)

    def _register_alias(self, node_name: str, alias: str) -> None:
        alias_text = self._first_text(alias)
        if not alias_text:
            return
        if alias_text not in self.alias_to_name:
            self.alias_to_name[alias_text] = node_name
        normalized = self._normalize_lookup_text(alias_text)
        if not normalized:
            return
        names = self.normalized_alias_to_names.setdefault(normalized, [])
        if node_name not in names:
            names.append(node_name)
        entry = (node_name, alias_text, normalized)
        if entry not in self.alias_entries:
            self.alias_entries.append(entry)

    def _register_manual_aliases(self) -> None:
        for alias, node_name in self.MANUAL_ALIAS_HINTS.items():
            if node_name in self.nodes:
                self._register_alias(node_name, alias)

    def _normalize_alias_values(self, value: Any) -> list[str]:
        if isinstance(value, list):
            aliases: list[str] = []
            for item in value:
                alias_text = self._first_text(item)
                if alias_text:
                    aliases.append(alias_text)
            return aliases
        alias_text = self._first_text(value)
        return [alias_text] if alias_text else []

    def _generate_lookup_variants(self, text: str) -> list[str]:
        variants: list[str] = []
        raw_text = self._first_text(text)
        if not raw_text:
            return variants

        compact = self._normalize_lookup_text(raw_text)
        if compact:
            variants.append(compact)

        trimmed = raw_text
        for prefix in self.GENERIC_PREFIXES:
            if trimmed.startswith(prefix) and len(trimmed) > len(prefix) + 1:
                trimmed = trimmed[len(prefix) :]
        trimmed_compact = self._normalize_lookup_text(trimmed)
        if trimmed_compact and trimmed_compact not in variants:
            variants.append(trimmed_compact)

        for suffix in self.GENERIC_SUFFIXES:
            if trimmed.endswith(suffix) and len(trimmed) > len(suffix) + 1:
                candidate = self._normalize_lookup_text(trimmed[: -len(suffix)])
                if candidate and candidate not in variants:
                    variants.append(candidate)

        for suffix in self.GENERIC_SUFFIXES:
            compact_suffix = self._normalize_lookup_text(suffix)
            if compact_suffix and compact.endswith(compact_suffix) and len(compact) > len(compact_suffix) + 1:
                candidate = compact[: -len(compact_suffix)]
                if candidate and candidate not in variants:
                    variants.append(candidate)
        return variants

    def _select_best_candidate(self, raw_name: str, candidates: list[str]) -> str:
        best_name = candidates[0]
        best_score = -1.0
        for candidate in candidates:
            score = self._score_candidate(raw_name, candidate, self._normalize_lookup_text(candidate))
            if score > best_score:
                best_score = score
                best_name = candidate
        return best_name

    def _score_candidate(self, raw_name: str, alias_text: str, alias_normalized: str) -> float:
        query_variants = self._generate_lookup_variants(raw_name)
        best_score = 0.0
        for query in query_variants:
            if not query or not alias_normalized:
                continue
            if query == alias_normalized:
                return 1.0
            score = SequenceMatcher(None, query, alias_normalized).ratio()
            if query in alias_normalized or alias_normalized in query:
                shorter = min(len(query), len(alias_normalized))
                longer = max(len(query), len(alias_normalized))
                score = max(score, 0.84 + 0.12 * (shorter / max(longer, 1)))
            overlap = self._character_overlap_ratio(query, alias_normalized)
            score = max(score, overlap)
            best_score = max(best_score, score)
        if best_score < 0.55:
            return 0.0
        if alias_text in self.nodes and self._normalize_lookup_text(alias_text) == self._normalize_lookup_text(raw_name):
            return max(best_score, 0.99)
        return best_score

    @staticmethod
    def _normalize_lookup_text(value: str) -> str:
        text = value.strip()
        if not text:
            return ""
        text = re.sub(r"[\s\-\_\(\)\[\]{}<>《》“”\"'‘’，,。；：:、/\\|？?！!·]+", "", text)
        return text.lower()

    @staticmethod
    def _character_overlap_ratio(left: str, right: str) -> float:
        left_chars = set(left)
        right_chars = set(right)
        if not left_chars or not right_chars:
            return 0.0
        intersection = len(left_chars & right_chars)
        union = len(left_chars | right_chars)
        if union == 0:
            return 0.0
        return intersection / union

    @staticmethod
    def _first_text(*values: Any) -> str:
        for value in values:
            if value is None:
                continue
            if isinstance(value, str):
                text = value.strip()
                if text:
                    return text
            elif isinstance(value, (int, float)):
                return str(value)
        return ""
