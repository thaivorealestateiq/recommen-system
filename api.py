from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from loguru import logger
from utils import get_recommendations
app = FastAPI()
security = HTTPBasic()


@app.post("/detection")
async def recommendationSearch(
    title: str
):
    print(title)
    # Get recommendations for the input product title
    recommendations_kmeans = get_recommendations(title)    

    return recommendations_kmeans
# Please read more about this here


@app.post("/detection-users")
async def recommendationUser(
    userid: int
):
    # Get recommendations for the input product title
    pass

# Please read more about this here
