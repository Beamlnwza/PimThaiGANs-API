from pydantic import BaseModel, Field, AnyHttpUrl, ValidationError, validator
from uuid import UUID, uuid4
from enum import Enum


class Empty(BaseModel):
    """
    Represents an empty object.

    This class is used to define an empty object with no additional fields.

    Attributes:
        Config (class): Configuration options for the Empty class.

    """

    class Config:
        extra = "forbid"


class ImageUrls(BaseModel):
    """
    Represents a model for storing image URLs.

    Attributes:
        image_urls (dict[str, AnyHttpUrl] | Empty): A dictionary containing image URLs.
            The keys represent the index of the image, and the values represent the URL of the image.
    """

    image_urls: dict[str, AnyHttpUrl] | Empty = Field(
        ...,
        example={
            "0": "https://example.com/image.png",
            "1": "https://example.com/image2.png",
        },
    )

    class Config:
        extra = "forbid"


class ImageReturn(BaseModel):
    """
    Represents the response object containing image information.

    Attributes:
        user_uuid (UUID): The UUID of the user.
        request_id (UUID): The UUID of the request.
        image_urls (dict[str, AnyHttpUrl]): A dictionary containing image URLs.
    """

    user_uuid: UUID = Field(..., example=uuid4())
    request_id: UUID = Field(..., example=uuid4())
    image_urls: dict[str, AnyHttpUrl]

    class Config:
        extra = "forbid"

    @classmethod
    def parse_obj(cls, obj):
        if "image_urls" in obj and isinstance(obj["image_urls"], dict):
            obj["image_urls"] = ImageUrls(**obj["image_urls"]).image_urls
        return super().parse_obj(obj)


from enum import Enum


class GvType(str, Enum):
    """
    Enumeration representing different types of Gv.
    """

    all = "all"
    index = "index"
    list = "list"
    current = "current"


class Request(BaseModel):
    uuid: UUID
    type: GvType
    index: int | None = None


class LeftReturn(BaseModel):
    user_uuid: UUID = Field(..., example=uuid4())
    request_id: UUID = Field(..., example=uuid4())
    index_left: list[int]
    missing_index: list[int]
