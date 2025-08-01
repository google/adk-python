from fastapi import APIRouter
router = APIRouter()
@router.get("/apihub")
async def get_apihub():
    return {"message": "API Hub endpoint"}