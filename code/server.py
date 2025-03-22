from fastapi import FastAPI
# from awb.api import router 
from awb_openai.api import router

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)