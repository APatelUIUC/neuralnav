import type { AblationResult, DecomposeResponse } from "../store/store";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8080";

/**
 * Send an activation vector to the backend for SAE feature decomposition.
 */
export async function decompose(
  activation: Float32Array,
  layer: number,
): Promise<DecomposeResponse> {
  const response = await fetch(`${BASE_URL}/decompose`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      activation: Array.from(activation),
      layer,
    }),
  });

  if (!response.ok) {
    throw new Error(`Decompose request failed: ${response.status}`);
  }

  return await response.json();
}

/**
 * Request an ablation (zero out one SAE feature) and get logit diff.
 */
export async function ablate(
  activation: Float32Array,
  layer: number,
  featureIndex: number,
  featureActivation: number,
  featureLabel: string,
): Promise<AblationResult> {
  const response = await fetch(`${BASE_URL}/ablate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      activation: Array.from(activation),
      layer,
      feature_index: featureIndex,
      feature_activation: featureActivation,
    }),
  });

  if (!response.ok) {
    throw new Error(`Ablate request failed: ${response.status}`);
  }

  const data = await response.json();

  return {
    featureIndex,
    featureLabel,
    tokensPromoted: (data.logit_diff_approx.tokens_promoted ?? []).map(
      (t: { token: string; logit_change: number }) => ({
        token: t.token,
        logitChange: t.logit_change,
      }),
    ),
    tokensDemoted: (data.logit_diff_approx.tokens_demoted ?? []).map(
      (t: { token: string; logit_change: number }) => ({
        token: t.token,
        logitChange: t.logit_change,
      }),
    ),
  };
}
