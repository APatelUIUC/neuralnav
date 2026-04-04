import { create } from "zustand";

export interface LogitEntry {
  token: string;
  logit: number;
}

export interface Feature {
  index: number;
  label: string;
  activation: number;
  top_positive_logits: LogitEntry[];
  top_negative_logits: LogitEntry[];
}

export interface DecomposeResponse {
  features: Feature[];
  reconstruction_score: number;
  total_features_active: number;
}

export interface AblationResult {
  featureIndex: number;
  featureLabel: string;
  tokensPromoted: { token: string; logitChange: number }[];
  tokensDemoted: { token: string; logitChange: number }[];
}

export interface AppState {
  prompt: string;
  tokens: string[];
  tokenIds: number[];
  logits: Float32Array | null;
  residuals: Float32Array[] | null;
  norms: number[][] | null;
  selectedToken: number | null;
  selectedLayer: number | null;
  features: Feature[] | null;
  featuresLoading: boolean;
  reconstructionScore: number | null;
  ablation: AblationResult | null;
  ablationLoading: boolean;
  modelReady: boolean;
  modelProgress: number;
  inferenceRunning: boolean;
  error: string | null;

  setPrompt: (prompt: string) => void;
  setTokens: (tokens: string[], tokenIds: number[]) => void;
  setInferenceResult: (
    logits: Float32Array,
    residuals: Float32Array[],
    norms: number[][],
  ) => void;
  selectCell: (tokenIdx: number | null, layerIdx: number | null) => void;
  setFeatures: (
    features: Feature[],
    reconstructionScore: number,
  ) => void;
  setFeaturesLoading: (loading: boolean) => void;
  setAblation: (result: AblationResult | null) => void;
  setAblationLoading: (loading: boolean) => void;
  setModelReady: (ready: boolean) => void;
  setModelProgress: (progress: number) => void;
  setInferenceRunning: (running: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

const initialState = {
  prompt: "",
  tokens: [],
  tokenIds: [],
  logits: null,
  residuals: null,
  norms: null,
  selectedToken: null,
  selectedLayer: null,
  features: null,
  featuresLoading: false,
  reconstructionScore: null,
  ablation: null,
  ablationLoading: false,
  modelReady: false,
  modelProgress: 0,
  inferenceRunning: false,
  error: null,
};

export const useStore = create<AppState>((set) => ({
  ...initialState,

  setPrompt: (prompt) => set({ prompt }),

  setTokens: (tokens, tokenIds) => set({ tokens, tokenIds }),

  setInferenceResult: (logits, residuals, norms) =>
    set({
      logits,
      residuals,
      norms,
      inferenceRunning: false,
      selectedToken: null,
      selectedLayer: null,
      features: null,
      ablation: null,
    }),

  selectCell: (tokenIdx, layerIdx) =>
    set({
      selectedToken: tokenIdx,
      selectedLayer: layerIdx,
      features: null,
      reconstructionScore: null,
      ablation: null,
    }),

  setFeatures: (features, reconstructionScore) =>
    set({ features, reconstructionScore, featuresLoading: false }),

  setFeaturesLoading: (featuresLoading) => set({ featuresLoading }),

  setAblation: (ablation) => set({ ablation, ablationLoading: false }),

  setAblationLoading: (ablationLoading) => set({ ablationLoading }),

  setModelReady: (modelReady) => set({ modelReady }),

  setModelProgress: (modelProgress) => set({ modelProgress }),

  setInferenceRunning: (inferenceRunning) => set({ inferenceRunning }),

  setError: (error) => set({ error }),

  reset: () => set(initialState),
}));
