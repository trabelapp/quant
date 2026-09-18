from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index():
    return "<!doctype html><html><head><meta charset=\"utf-8\"></head><body></body></html>"
