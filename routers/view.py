from fastapi import APIRouter
from pydantic import UUID4

router = APIRouter(
    prefix="/v",
    tags=["view"],
    responses={404: {"description": "Not found"}},
)


@router.get("/info")
async def root():
    return {"info": "for return view image"}


@router.get("/")
async def root(uuid: UUID4):
    return {"info": "Welcome to pimthaigans api", "contact": "github.com/Beamlnwza"}
