from fastapi import APIRouter
router = APIRouter()
@router.get("/openapi_tools")
async def get_openapi_tools():
    return {"message": "OpenAPI tools endpoint"}