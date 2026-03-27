from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.backend.settings import Settings, get_settings
from app.backend.services.artifact_store import ArtifactPaths
from app.backend.services.loader import ArtifactLoader
from app.backend.services.predictor import PredictorService
from app.backend.services.agent_service import AgentService
from app.backend.services.llm_service import LLMService

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentChatMessage(BaseModel):
    role: str
    content: str


class AgentChatRequest(BaseModel):
    message: str = Field(..., description="User message to the procurement agent")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    history: Optional[List[AgentChatMessage]] = Field(default_factory=list)


class AgentChatResponse(BaseModel):
    answer: str
    detected_intent: str
    used_tools: List[str]
    parsed_context: Dict[str, Any]
    data: Dict[str, Any]


def _get_agent_service(settings: Settings = Depends(get_settings)) -> AgentService:
    paths = ArtifactPaths(root=settings.artifacts_dir)
    loader = ArtifactLoader(paths)
    predictor = PredictorService(loader)
    llm = LLMService(settings)
    return AgentService(loader, predictor, llm=llm)


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(
    payload: AgentChatRequest,
    svc: AgentService = Depends(_get_agent_service),
) -> AgentChatResponse:
    try:
        history = [{"role": m.role, "content": m.content} for m in payload.history] if payload.history else []
        result = await svc.chat(
            message=payload.message,
            context=payload.context or {},
            history=history,
        )
        return AgentChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def agent_health():
    return {"status": "ok", "service": "agent"}