from jrjModelRegistry import handleDashboard, jrjRouterModelRegistry
from fastapi import FastAPI
import statsmodels.api as sm
import pandas as pd
from dotenv import load_dotenv

from pathlib import Path

env_path = Path(".env-live")


if env_path.exists():
    load_dotenv(dotenv_path=env_path)


class ModelData:
    pass


model = ModelData()


app = FastAPI()


app.include_router(jrjRouterModelRegistry)

handleDashboard(app)


@app.get("/")
async def root():
    return {"message": "Hello World"}
