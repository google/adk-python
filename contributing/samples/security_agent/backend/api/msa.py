from fastapi import APIRouter
router = APIRouter()
@router.get("/msa")
async def get_msa():
    return {"message": "MSA endpoint"}