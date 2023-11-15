from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import generate, view

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
    router_list = [generate, view]
    for router in router_list:
        app.include_router(router.router)


config_router()


@app.get("/")
async def root():
    return {"info": "Welcome to pimthaigans api", "contact": "github.com/Beamlnwza"}
