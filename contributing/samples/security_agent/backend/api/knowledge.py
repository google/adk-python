from fastapi import APIRouter
router = APIRouter()
@router.get("/knowledge")
async def get_knowledge():
    return {"message": "Knowledge endpoint"}