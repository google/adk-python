from fastapi import APIRouter
router = APIRouter()
@router.get("/incidents")
async def get_incidents():
    return {"message": "Incidents endpoint"}