from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Astris is online."}

@app.post("/chat")
def chat_endpoint():
    return {"response": "Councillor is ready."}