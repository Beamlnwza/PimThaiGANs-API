from fastapi import APIRouter
from uuid import UUID
from libs.type import ImageReturn
from uuid import uuid4
import json

router = APIRouter(
    prefix="/v",
    tags=["view"],
    responses={404: {"description": "Not found"}},
)


@router.get("/info")
async def root():
    return {"info": "for return view image"}


@router.get("/status")
async def root():
    return {"status": "ok"}


@router.get("/")
async def root(uuid: UUID) -> ImageReturn:
    request_id = uuid4()

    return {
        "user_uuid": uuid,
        "request_id": request_id,
        "image_urls": {
            "0": "https://example.com/image0.png",
            "1": "https://example.com/image1.png",
            "2": "https://example.com/image2.png",
        },
    }
