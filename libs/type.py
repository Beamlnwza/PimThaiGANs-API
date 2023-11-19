from pydantic import BaseModel, Field, AnyHttpUrl
from uuid import UUID


class ImageReturn(BaseModel):
    user_uuid: UUID = Field(..., example="123e4567-e89b-12d3-a456-426614174000")
    request_id: UUID = Field(..., example="123e4567-e89b-12d3-a456-426614174000")
    image_urls: dict[str, AnyHttpUrl] = Field(
        ...,
        example={
            "0": "https://example.com/image.png",
            "1": "https://example.com/image2.png",
        },
    )
