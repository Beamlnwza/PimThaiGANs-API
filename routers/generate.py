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
    """
    Generate an image based on the given UUID and type.

    Args:
        uuid (UUID): The UUID of the image.
        type (GvType): The type of the image.
        index (int | None, optional): The index of the image. Defaults to None.

    Returns:
        ImageReturn: The generated image.

    Raises:
        HTTPException: If the type or UUID is invalid.
    """
    request_id = uuid4()

    match type:
        case GvType.all:
            return g_all(uuid, request_id)
        case GvType.index:
            return g_index(uuid, request_id, index)

    raise HTTPException(status_code=404, detail="Invalid type nor uuid")


def g_all(uuid: UUID, request_id: UUID) -> ImageReturn:
    """
    Generate all images for the given UUID and request ID.

    Args:
        uuid (UUID): The UUID of the user.
        request_id (UUID): The UUID of the request.

    Returns:
        ImageReturn: A dictionary containing the user UUID, request ID, and image URLs.
    """
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
    """
    Generate and return an image based on the given index.

    Args:
        uuid (UUID): The UUID of the user.
        request_id (UUID): The UUID of the request.
        index (int): The index of the image to generate.

    Returns:
        ImageReturn: The generated image along with user UUID, request ID, and image URLs.

    Raises:
        HTTPException: If the index is out of range (not between 0 and 87).
    """
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
