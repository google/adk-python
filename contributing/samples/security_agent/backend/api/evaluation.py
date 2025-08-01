from fastapi import APIRouter
router = APIRouter()
@router.get("/evaluation")
async def get_evaluation():
    return {"message": "Evaluation endpoint"}