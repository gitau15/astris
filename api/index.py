from fastapi import FastAPI # pyright: ignore[reportMissingImports]

# Create the FastAPI app instance
app = FastAPI()

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Astris. The council is assembling."}

# Define the chat endpoint
@app.post("/chat")
def chat_endpoint():
    return {"response": "The Lead Councillor of Astris is ready."}