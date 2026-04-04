export interface LogitEntry {
  token: string;
  logit: number;
}

export interface FeatureInfo {
  index: number;
  activation: number;
  label: string;
  top_positive_logits: LogitEntry[];
  top_negative_logits: LogitEntry[];
}

export interface DecomposeRequest {
  activation: number[];
  layer: number;
}

export interface DecomposeResponse {
  features: FeatureInfo[];
  reconstruction_score: number;
  total_features_active: number;
}

export interface AblateRequest {
  activation: number[];
  layer: number;
  feature_index: number;
  feature_activation: number;
}

export interface AblateResponse {
  modified_activation: number[];
  logit_diff_approx: {
    tokens_promoted: { token: string; logit_change: number }[];
    tokens_demoted: { token: string; logit_change: number }[];
  };
}
