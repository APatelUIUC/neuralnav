"""API route definitions for SAE explorer."""

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .labels import get_label
from .logit_lens import compute_logit_diff, compute_logit_effects
from .sae import decode, encode, top_k_features

router = APIRouter()


# --- Request / Response models ---


class DecomposeRequest(BaseModel):
    activation: list[float] = Field(..., min_length=768, max_length=768)
    layer: int = Field(..., ge=0, le=11)


class LogitEntry(BaseModel):
    token: str
    logit: float


class FeatureInfo(BaseModel):
    index: int
    activation: float
    label: str
    top_positive_logits: list[LogitEntry]
    top_negative_logits: list[LogitEntry]


class DecomposeResponse(BaseModel):
    features: list[FeatureInfo]
    reconstruction_score: float
    total_features_active: int


class AblateRequest(BaseModel):
    activation: list[float] = Field(..., min_length=768, max_length=768)
    layer: int = Field(..., ge=0, le=11)
    feature_index: int = Field(..., ge=0)
    feature_activation: float


class AblateResponse(BaseModel):
    modified_activation: list[float]
    logit_diff_approx: dict


class ExplainRequest(BaseModel):
    activation: list[float] = Field(..., min_length=768, max_length=768)
    layer: int = Field(..., ge=0, le=11)


class ExplainResponse(BaseModel):
    description: str
    reconstruction_cosine: float


# --- Endpoints ---


@router.post("/decompose", response_model=DecomposeResponse)
async def decompose(req: DecomposeRequest, request: Request) -> DecomposeResponse:
    """Decompose an activation vector into its top SAE features."""
    layer_weights = request.app.state.layer_weights
    W_U = request.app.state.W_U

    if req.layer not in layer_weights:
        raise HTTPException(status_code=404, detail=f"Weights for layer {req.layer} not loaded")

    weights = layer_weights[req.layer]
    x = np.array(req.activation, dtype=np.float32)

    # Encode
    z = encode(x, weights)
    total_active = int(np.sum(z > 0))

    # Top-k features
    indices, values = top_k_features(z, k=20)

    if len(indices) == 0:
        return DecomposeResponse(
            features=[], reconstruction_score=0.0, total_features_active=0
        )

    # Reconstruction quality
    x_hat = decode(z, weights)
    cos_sim = _cosine_similarity(x, x_hat)

    # Logit effects
    logit_effects = compute_logit_effects(indices, values, weights.W_dec, W_U)

    # Build response
    features = []
    for i, (idx, val) in enumerate(zip(indices, values)):
        features.append(
            FeatureInfo(
                index=int(idx),
                activation=round(float(val), 4),
                label=get_label(req.layer, int(idx)),
                top_positive_logits=[
                    LogitEntry(**e) for e in logit_effects[i]["top_positive_logits"]
                ],
                top_negative_logits=[
                    LogitEntry(**e) for e in logit_effects[i]["top_negative_logits"]
                ],
            )
        )

    return DecomposeResponse(
        features=features,
        reconstruction_score=round(float(cos_sim), 4),
        total_features_active=total_active,
    )


@router.post("/ablate", response_model=AblateResponse)
async def ablate(req: AblateRequest, request: Request) -> AblateResponse:
    """Ablate a single feature and return the modified activation."""
    layer_weights = request.app.state.layer_weights
    W_U = request.app.state.W_U

    if req.layer not in layer_weights:
        raise HTTPException(status_code=404, detail=f"Weights for layer {req.layer} not loaded")

    weights = layer_weights[req.layer]

    if req.feature_index >= weights.W_dec.shape[0]:
        raise HTTPException(
            status_code=400,
            detail=f"Feature index {req.feature_index} out of range (max {weights.W_dec.shape[0] - 1})",
        )

    x = np.array(req.activation, dtype=np.float32)

    # Ablation: remove this feature's contribution
    modified = x - req.feature_activation * weights.W_dec[req.feature_index].astype(np.float32)

    # Logit diff approximation
    logit_diff = compute_logit_diff(
        feature_index=req.feature_index,
        feature_activation=req.feature_activation,
        W_dec=weights.W_dec,
        W_U=W_U,
    )

    return AblateResponse(
        modified_activation=[round(float(v), 6) for v in modified],
        logit_diff_approx=logit_diff,
    )


@router.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest, request: Request) -> ExplainResponse:
    """Verbalize an activation with the NLA and report reconstruction faithfulness."""
    nla = getattr(request.app.state, "nla", None)
    if nla is None:
        raise HTTPException(
            status_code=503,
            detail="NLA not loaded — train the verbalizer + reconstructor first "
                   "(scripts/nla/train_av.py, train_ar.py).",
        )
    from .nla import explain as run_explain

    description, cosine = run_explain(nla, req.activation)
    return ExplainResponse(description=description, reconstruction_cosine=round(cosine, 4))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
