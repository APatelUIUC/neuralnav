"""FastAPI application for SAE Explorer."""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .labels import load_labels
from .routes import router
from .sae import load_all_weights

DATA_DIR = os.environ.get("SAE_DATA_DIR", "data")

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "https://sae-explorer.vercel.app",
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load SAE weights and feature labels into memory at startup."""
    print(f"Loading SAE weights from {DATA_DIR}...")
    layer_weights, W_U = load_all_weights(DATA_DIR)
    print(f"Loaded weights for {len(layer_weights)} layers, W_U shape: {W_U.shape}")

    load_labels(DATA_DIR)
    print("Feature labels loaded.")

    app.state.layer_weights = layer_weights
    app.state.W_U = W_U

    # NLA (optional): loads only if trained models + torch are present.
    app.state.nla = None
    try:
        from pathlib import Path

        from .nla import load_nla

        nla_out = Path(os.environ.get(
            "NLA_OUT_DIR", Path(__file__).resolve().parents[3] / "scripts" / "nla" / "out"))
        app.state.nla = load_nla(nla_out / "av", nla_out / "ar.pt")
        print("NLA loaded — /explain enabled" if app.state.nla
              else "NLA models not found — /explain disabled until trained")
    except ImportError:
        print("torch/transformers not installed in server env — /explain disabled")

    yield


app = FastAPI(
    title="SAE Explorer API",
    description="Mechanistic interpretability tool for Sparse Autoencoder feature exploration.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
