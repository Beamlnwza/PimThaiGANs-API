from fastapi import APIRouter

router = APIRouter(
    prefix="/g",
    tags=["generate"],
    responses={404: {"description": "Not found"}},
)


@router.get("/info")
async def root():
    return {"info": "for generate image"}
