"""Agent API endpoints for ADK integration."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class ChatRequest(BaseModel):
    """Request model for agent chat."""
    query: str
    user_id: Optional[str] = "default_user"


class ChatResponse(BaseModel):
    """Response model for agent chat."""
    success: bool
    response: str
    user_id: str


@router.get("/")
async def get_agent_info(request: Request):
    """Get agent information and capabilities."""
    try:
        agent_service = request.app.state.agent_service
        agent_info = agent_service.get_agent_info()
        
        return {
            "success": True,
            "agent_info": agent_info,
            "available_tools": agent_service.get_agent_tools(),
            "status": "ready"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent info: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(chat_request: ChatRequest, request: Request):
    """Chat with the ADK security agent."""
    try:
        agent_service = request.app.state.agent_service
        
        # Get response from agent service
        response = await agent_service.chat(
            message=chat_request.query,
            user_id=chat_request.user_id
        )
        
        return ChatResponse(
            success=True,
            response=response,
            user_id=chat_request.user_id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to chat with agent: {str(e)}"
        )


@router.post("/query")
async def query_agent(chat_request: ChatRequest, request: Request):
    """Send a query to the agent (alternative endpoint)."""
    try:
        agent_service = request.app.state.agent_service
        
        # Get response from agent service
        response = await agent_service.query_agent(
            query=chat_request.query,
            user_id=chat_request.user_id
        )
        
        return {
            "success": True,
            "response": response,
            "user_id": chat_request.user_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to query agent: {str(e)}"
        )


@router.delete("/session/{user_id}")
async def close_session(user_id: str, request: Request):
    """Close a user session."""
    try:
        agent_service = request.app.state.agent_service
        await agent_service.close_session(user_id)
        
        return {
            "success": True,
            "message": f"Session closed for user: {user_id}"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to close session: {str(e)}"
        )