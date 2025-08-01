from fastapi import APIRouter
router = APIRouter()
@router.get("/documentation")
async def get_documentation():
    return {"message": "Documentation endpoint"}