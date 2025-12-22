from fastapi import FastAPI
from infer import infer

app = FastAPI()

@app.get("/predict/{idx}")
def predict(idx: int):
    return infer(idx)
