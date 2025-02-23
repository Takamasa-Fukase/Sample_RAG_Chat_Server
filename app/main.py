from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "どうも、ウルトラ深瀬です"}
