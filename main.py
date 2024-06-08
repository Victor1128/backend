from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pydantic import BaseModel
import uvicorn
import pandas as pd
import sys
import os

from model import Model
from update_news import update_news

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))


class Text(BaseModel):
    title: str
    content: str


model = Model('models')

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def update_news_task():
    while True:
        update_news()
        await asyncio.sleep(5 * 60 * 60)


@app.on_event("startup")
def on_startup():
    asyncio.create_task(update_news_task())


@app.get("/")
async def root():
    return {"message": model.get()}


@app.get("/news")
async def news():
    try:
        df = pd.read_csv('satire.csv')
    except FileNotFoundError:
        return {
            "news": []
        }
    return {
        "news": df.to_dict(orient='records')
    }

@app.post("/predict")
async def predict(text: Text):
    df = pd.DataFrame({
        'title': [text.title],
        'content': [text.content]
    })
    model.load_data(df)
    predictions = model.predict()
    print(predictions)
    return {
        "predictions": predictions
    }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
