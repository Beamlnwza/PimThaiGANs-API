from typing import Union
from fastapi import APIRouter, HTTPException
from uuid import UUID, uuid4
from libs.type import ImageReturn, GvType
from libs.s3 import s3_resource

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
async def root(uuid: UUID, type: GvType, index: int | None = None) -> ImageReturn:
    request_id = uuid4()

    match type:
        case GvType.all:
            return v_all(uuid, request_id)
        case GvType.index:
            return v_index(uuid, request_id, index)

    raise HTTPException(status_code=404, detail="Invalid type nor uuid")


def v_all(uuid: UUID, request_id: UUID) -> ImageReturn:
    return {
        "user_uuid": uuid,
        "request_id": request_id,
        "image_urls": {
            "0": "https://example.com/image0.png",
            "1": "https://example.com/image1.png",
            "2": "https://example.com/image2.png",
        },
    }


def v_index(uuid: UUID, request_id: UUID, index: int) -> ImageReturn:
    if index not in range(0, 88):
        raise HTTPException(status_code=404, detail="Class out of index")

    return {
        "user_uuid": uuid,
        "request_id": request_id,
        "image_urls": {
            f"{index}": "https://example.com/image0.png",
        },
    }
