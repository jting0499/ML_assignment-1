from dotenv import load_dotenv

from pathlib import Path

env_path = Path(".env-live")


import pandas as pd
import statsmodels.api as sm

if env_path.exists():
    load_dotenv(dotenv_path=env_path)

class ModelData: pass

model = ModelData()


from fastapi import FastAPI

from jrjModelRegistry import handleDashboard, jrjRouterModelRegistry


app = FastAPI()



app.include_router(jrjRouterModelRegistry)

handleDashboard(app)

@app.get("/")
async def root():
    return {"message": "Hello World"}