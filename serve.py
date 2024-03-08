from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from main import chain1,chain2

app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(app,chain1,path="/search-ollama")
add_routes(app,chain2,path="/search-gemini")

if __name__=="__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)