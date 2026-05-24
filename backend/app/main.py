from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router


app = FastAPI(
    title="Predicting Missing Examination Component Scores",
    version="1.0.0",
    description="Privacy-preserving, explainable ML capstone application for examination component score prediction.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
