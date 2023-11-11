from fastapi import APIRouter

router = APIRouter(
    prefix="/generate",
    tags=["generate"],
    responses={404: {"description": "Not found"}},
)


@router.get("/")
async def root():
    return {"message": "Hello World 2"}
