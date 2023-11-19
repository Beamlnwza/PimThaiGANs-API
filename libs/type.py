from pydantic import BaseModel, Field, AnyHttpUrl, ValidationError, validator
from uuid import UUID, uuid4
from enum import Enum


class Empty(BaseModel):
    class Config:
        extra = "forbid"


class ImageUrls(BaseModel):
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


class GvType(str, Enum):
    all = "all"
    index = "index"
    list = "list"
    current = "current"
