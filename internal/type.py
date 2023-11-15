from pydantic import BaseModel, Field, UUID4, AnyHttpUrl


class ImageReturn(BaseModel):
    requestid: UUID4 = Field(..., example="123e4567-e89b-12d3-a456-426614174000")
    useruuid: UUID4 = Field(..., example="123e4567-e89b-12d3-a456-426614174000")


class ImageList(BaseModel):
    imageUrls: list[AnyHttpUrl] = Field(
        ..., example=["https://example.com/image.jpg", "https://example.com/image2.jpg"]
    )
