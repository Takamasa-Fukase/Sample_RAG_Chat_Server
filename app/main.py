import json, threading
from typing import List, Optional
from fastapi import FastAPI, Request, HTTPException
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

app = FastAPI()

# CORSを回避するために追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/ping')
def ping():
    return {'data': {'message': 'OK'}}

# @app.post('/chat')
# def get_answer():