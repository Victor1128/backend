from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from model import Model


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

@app.get("/")
async def root():
    return {"message": model.get()}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/predict")
async def predict(text: Text):
    model.load_data([text.title], [text.content])
    predictions = model.predict()
    print(predictions)
    return {
        "predictions": predictions
    }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
