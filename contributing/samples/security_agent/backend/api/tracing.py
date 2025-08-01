from fastapi import APIRouter
router = APIRouter()
@router.get("/tracing")
async def get_tracing():
    return {"message": "Tracing endpoint"}