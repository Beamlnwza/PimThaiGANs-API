from fastapi import APIRouter, HTTPException
from uuid import UUID, uuid4

from libs.type import ImageReturn, GvType
from libs.models import generate_image_index
from libs.image import extract_image, image_byte

router = APIRouter(
    prefix="/g",
    tags=["generate"],
    responses={404: {"description": "Not found"}},
)


@router.get("/info")
async def root():
    return {"info": "for generate image"}


@router.get("/")
async def root(uuid: UUID, type: GvType, index: int | None = None) -> ImageReturn:
    request_id = uuid4()

    match type:
        case GvType.all:
            return g_all(uuid, request_id)
        case GvType.index:
            return g_index(uuid, request_id, index)

    raise HTTPException(status_code=404, detail="Invalid type nor uuid")


def g_all(uuid: UUID, request_id: UUID) -> ImageReturn:
    img_urls = {}

    for index in range(0, 88):
        img = generate_image_index(index)
        img = extract_image(img)

        image_byte(uuid, img_urls, index, img)

    return {
        "user_uuid": uuid,
        "request_id": request_id,
        "image_urls": img_urls,
    }


def g_index(uuid: UUID, request_id: UUID, index: int) -> ImageReturn:
    if index not in range(0, 88):
        raise HTTPException(status_code=404, detail="Class out of index")

    img_urls = {}

    img = generate_image_index(index)
    img = extract_image(img)

    image_byte(uuid, img_urls, index, img)

    return {
        "user_uuid": uuid,
        "request_id": request_id,
        "image_urls": img_urls,
    }
