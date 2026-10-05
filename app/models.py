from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class OCRQuestion(BaseModel):
    question_text: str
    knowledge_points: list[str] = Field(default_factory=list)
    student_answer: str
    correct_answer: str
    error_steps: list[str] = Field(default_factory=list)
    handwriting_boxes: list[list[int]] = Field(default_factory=list)


class OCRResult(BaseModel):
    provider: str
    image_name: str
    subject: str = ""
    grade: str = ""
    questions: list[OCRQuestion] = Field(default_factory=list)
    raw_summary: str = ""


class ConfidenceItem(BaseModel):
    knowledge_point: str
    confidence: float
    reason: str


class GraphNode(BaseModel):
    name: str
    subject: str = ""
    node_type: str = ""
    role: str = ""
    layer: int = 0
    description: str = ""


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    evidence: str = ""


class GraphSnapshot(BaseModel):
    focus_node: str = ""
    prerequisite_gap: str = ""
    node_count: int = 0
    edge_count: int = 0
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class GraphCandidate(BaseModel):
    name: str
    subject: str = ""
    node_type: str = "Knowledge"
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    related_to: list[str] = Field(default_factory=list)
    is_a: list[str] = Field(default_factory=list)
    similar_existing_nodes: list[str] = Field(default_factory=list)
    source: str = "heuristic"


class GraphSuggestionRequest(BaseModel):
    question_text: str = ""
    subject: str = ""
    grade: str = ""
    knowledge_points: list[str] = Field(default_factory=list)
    llm_provider: str = ""
    llm_model: str = ""
    ollama_base_url: str = ""
    deepseek_base_url: str = ""


class GraphSuggestionResponse(BaseModel):
    missing_knowledge_points: list[str] = Field(default_factory=list)
    suggestions: list[GraphCandidate] = Field(default_factory=list)
    message: str = ""


class GraphAdditionRequest(BaseModel):
    suggestions: list[GraphCandidate] = Field(default_factory=list)
    focus_knowledge_points: list[str] = Field(default_factory=list)
    prerequisite_gap: str = ""


class GraphAdditionResponse(BaseModel):
    added_nodes: list[str] = Field(default_factory=list)
    focus_node: str = ""
    graph_paths: list[list[str]] = Field(default_factory=list)
    graph_snapshot: GraphSnapshot = Field(default_factory=GraphSnapshot)
    message: str = ""


class DiagnosticResult(BaseModel):
    provider: str
    error_level: str
    prerequisite_gap: str
    confidence_distribution: list[ConfidenceItem] = Field(default_factory=list)
    advice: str
    evidence: list[str] = Field(default_factory=list)
    graph_paths: list[list[str]] = Field(default_factory=list)
    graph_snapshot: GraphSnapshot = Field(default_factory=GraphSnapshot)
    prompt_used: str = ""
    used_fallback: bool = False
    llm_error: str = ""


class SceneRecipe(BaseModel):
    scene_type: str
    title: str
    description: str
    steps: list[str] = Field(default_factory=list)
    controls: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class InterventionResult(BaseModel):
    provider: str
    prompt: str
    generation_status: str
    model_download_url: Optional[str] = None
    preview_image_url: Optional[str] = None
    scene_recipe: SceneRecipe


class TrendItem(BaseModel):
    knowledge_point: str
    count: int


class HistorySummary(BaseModel):
    total_records: int
    top_blind_spots: list[TrendItem] = Field(default_factory=list)


class ProcessStep(BaseModel):
    key: str
    label: str
    status: str
    detail: str = ""


class PipelineResult(BaseModel):
    analysis_id: str
    ocr_result: OCRResult
    diagnostic_result: DiagnosticResult
    intervention_result: InterventionResult
    history_summary: HistorySummary
    process_steps: list[ProcessStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
