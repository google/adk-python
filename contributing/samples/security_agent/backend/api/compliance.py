from fastapi import APIRouter
router = APIRouter()
@router.get("/compliance")
async def get_compliance():
    return {"message": "Compliance endpoint"}