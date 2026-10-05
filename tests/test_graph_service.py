from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.graph_service import KnowledgeGraphService  # noqa: E402


def test_graph_service_supports_k12_kgraph_edge_format(tmp_path: Path) -> None:
    graph_path = tmp_path / "k12_kgraph.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "c1", "type": "Concept", "name": "分数意义", "subject": "数学"},
                    {"id": "c2", "type": "Concept", "name": "最小公倍数", "subject": "数学"},
                    {"id": "s1", "type": "Skill", "name": "分数通分", "subject": "数学"},
                    {"id": "s2", "type": "Skill", "name": "分数加法", "subject": "数学"},
                ],
                "edges": [
                    {"source": "c1", "target": "s1", "relation": "prerequisites_for"},
                    {"source": "c2", "target": "s1", "relation": "prerequisites_for"},
                    {"source": "s1", "target": "s2", "relation": "prerequisites_for"},
                    {"source": "c1", "target": "c2", "relation": "relates_to"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    service = KnowledgeGraphService(graph_path)

    assert service.get_prerequisites("分数通分") == ["分数意义", "最小公倍数"]
    assert service.get_all_paths("分数加法") == [
        ["分数意义", "分数通分", "分数加法"],
        ["最小公倍数", "分数通分", "分数加法"],
    ]
    assert service.get_lowest_prerequisite("分数加法") == "分数意义"


def test_repo_merged_graph_uses_k12_kgraph_and_project_supplements() -> None:
    graph_path = PROJECT_ROOT / "app" / "data" / "knowledge_graph_k12_merged.json"
    service = KnowledgeGraphService(graph_path)

    assert "四舍五入法" in service.get_prerequisites("近似数")
    assert "分数通分" in service.get_prerequisites("分数加法")
    assert "磁场方向" in service.get_prerequisites("电磁场方向判断")


def test_graph_service_builds_richer_learning_snapshot() -> None:
    graph_path = PROJECT_ROOT / "app" / "data" / "knowledge_graph_k12_merged.json"
    service = KnowledgeGraphService(graph_path)

    snapshot = service.build_learning_snapshot(
        focus_name="电磁场方向判断",
        prerequisite_gap="磁感线",
    )

    node_names = {item["name"] for item in snapshot["nodes"]}
    relations = {(item["source"], item["target"], item["relation"]) for item in snapshot["edges"]}

    assert snapshot["focus_node"] == "电磁场方向判断"
    assert snapshot["prerequisite_gap"] == "磁感线"
    assert "磁场方向" in node_names
    assert "空间想象力" in node_names
    assert ("磁感线", "磁场方向", "prerequisites_for") in relations
    assert ("磁场方向", "电磁场方向判断", "prerequisites_for") in relations


def test_graph_service_normalizes_alias_and_fuzzy_knowledge_point_names() -> None:
    graph_path = PROJECT_ROOT / "app" / "data" / "knowledge_graph_k12_merged.json"
    service = KnowledgeGraphService(graph_path)

    assert service.normalize_node_name("分数通分应用题") == "分数通分"
    assert service.normalize_node_name("最小公倍数问题") == "最小公倍数"
    assert service.normalize_node_name("磁感线方向判断") == "磁场方向"
    assert service.get_all_paths("分数加法相关题") == [
        ["分数意义", "分数通分", "分数加法"],
        ["最小公倍数", "分数通分", "分数加法"],
    ]


def test_graph_service_can_persist_new_graph_candidates(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"name": "数列基础", "subject": "数学"},
                    {"name": "代数式化简", "subject": "数学"},
                ],
                "edges": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    service = KnowledgeGraphService(graph_path)
    added = service.add_graph_candidates(
        [
            {
                "name": "数列的递推规律",
                "subject": "数学",
                "description": "理解递推关系并据此推导通项或求和结论。",
                "aliases": ["兔子数列"],
                "prerequisites": ["数列基础"],
                "related_to": ["代数式化简"],
                "is_a": [],
            }
        ]
    )

    assert added == ["数列的递推规律"]
    reloaded = KnowledgeGraphService(graph_path)
    assert reloaded.normalize_node_name("兔子数列") == "数列的递推规律"
    assert reloaded.get_prerequisites("数列的递推规律") == ["数列基础"]
    snapshot = reloaded.build_learning_snapshot(focus_name="数列的递推规律", prerequisite_gap="数列基础")
    edge_keys = {(item["source"], item["target"], item["relation"]) for item in snapshot["edges"]}
    assert ("数列基础", "数列的递推规律", "prerequisites_for") in edge_keys
    assert ("数列的递推规律", "代数式化简", "relates_to") in edge_keys
