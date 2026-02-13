from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

def main():
    uvicorn.run("variable_importance.main:app", host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
