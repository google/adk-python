from fastapi import APIRouter
router = APIRouter()
@router.get("/agent")
async def get_agent():
    return {"message": "Agent endpoint"}