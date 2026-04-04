import { useEffect, useRef, useCallback } from "react";
import { useStore } from "../store/store";
import { extractActivation } from "../lib/activations";
import { decompose, ablate } from "../lib/api";

export function useSAE() {
  const selectedToken = useStore((s) => s.selectedToken);
  const selectedLayer = useStore((s) => s.selectedLayer);
  const residuals = useStore((s) => s.residuals);
  const setFeatures = useStore((s) => s.setFeatures);
  const setFeaturesLoading = useStore((s) => s.setFeaturesLoading);
  const setAblation = useStore((s) => s.setAblation);
  const setAblationLoading = useStore((s) => s.setAblationLoading);
  const setError = useStore((s) => s.setError);

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Watch for cell selection changes -> trigger decompose
  useEffect(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    if (
      selectedToken === null ||
      selectedLayer === null ||
      !residuals
    ) {
      return;
    }

    setFeaturesLoading(true);

    debounceRef.current = setTimeout(async () => {
      try {
        const activation = extractActivation(
          residuals,
          selectedLayer,
          selectedToken,
        );
        const result = await decompose(activation, selectedLayer);
        setFeatures(result.features, result.reconstruction_score);
      } catch (err) {
        setError(
          `Feature decomposition failed: ${err instanceof Error ? err.message : String(err)}`,
        );
        setFeaturesLoading(false);
      }
    }, 300);

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, [selectedToken, selectedLayer, residuals, setFeatures, setFeaturesLoading, setError]);

  const runAblation = useCallback(
    async (featureIndex: number, featureActivation: number, featureLabel: string) => {
      if (
        selectedToken === null ||
        selectedLayer === null ||
        !residuals
      ) {
        return;
      }

      try {
        setAblationLoading(true);
        const activation = extractActivation(
          residuals,
          selectedLayer,
          selectedToken,
        );
        const result = await ablate(
          activation,
          selectedLayer,
          featureIndex,
          featureActivation,
          featureLabel,
        );
        setAblation(result);
      } catch (err) {
        setError(
          `Ablation failed: ${err instanceof Error ? err.message : String(err)}`,
        );
        setAblationLoading(false);
      }
    },
    [selectedToken, selectedLayer, residuals, setAblation, setAblationLoading, setError],
  );

  const resetAblation = useCallback(() => {
    setAblation(null);
  }, [setAblation]);

  return { runAblation, resetAblation };
}
