from fastapi import APIRouter
from uuid import UUID, uuid4

from libs.type import LeftReturn
from libs.s3 import s3_resource

router = APIRouter(
    prefix="/v",
    tags=["view"],
    responses={404: {"description": "Not found"}},
)


@router.get("/info")
async def root():
    return {"info": "for return view image"}


@router.get("/")
async def root(uuid: UUID) -> LeftReturn:
    """
    Retrieves the root information for a given UUID.

    Args:
        uuid (UUID): The UUID of the user.

    Returns:
        dict: A dictionary containing the user UUID, request ID, index left, and missing index.
    """
    request_id = uuid4()

    resource = s3_resource()
    bucket = resource.Bucket("pimthaigans")
    objects = bucket.objects.filter(Prefix=f"images/{uuid}/")

    index_left = []
    for obj in objects:
        index = obj.key.split("/")[2].split(".")[0]
        index_left.append(int(index))

    missing_index = list(set(range(0, 88)) - set(index_left))

    return {
        "user_uuid": uuid,
        "request_id": request_id,
        "index_left": index_left,
        "missing_index": missing_index,
    }
