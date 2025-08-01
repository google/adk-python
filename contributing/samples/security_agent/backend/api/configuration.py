from fastapi import APIRouter
router = APIRouter()
@router.get("/configuration")
async def get_configuration():
    return {"message": "Configuration endpoint"}