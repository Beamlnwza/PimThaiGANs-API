from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import generate

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


def config_router():
    app.include_router(generate.router)


config_router()


@app.get("/")
async def root():
    return {"message": "Hello World"}
