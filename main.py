from fastapi import FastAPI
import os

app = FastAPI()


@app.get("/")
async def main():
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run("main:app", host=os.environ["HOST"], port=os.environ["PORT"])