import { useCallback, useEffect, useRef } from "react";
import { initSession, runInference } from "../lib/onnx";
import { computeNorms } from "../lib/activations";
import { useTokenizer } from "./useTokenizer";
import { useStore } from "../store/store";

export function useGPT2() {
  const sessionRef = useRef<Awaited<ReturnType<typeof initSession>> | null>(
    null,
  );
  const initializingRef = useRef(false);

  const { encode, tokenize } = useTokenizer();

  const setModelReady = useStore((s) => s.setModelReady);
  const setModelProgress = useStore((s) => s.setModelProgress);
  const setInferenceRunning = useStore((s) => s.setInferenceRunning);
  const setTokens = useStore((s) => s.setTokens);
  const setInferenceResult = useStore((s) => s.setInferenceResult);
  const setError = useStore((s) => s.setError);
  const setPrompt = useStore((s) => s.setPrompt);

  // Initialize ONNX session on mount
  useEffect(() => {
    if (initializingRef.current || sessionRef.current) return;
    initializingRef.current = true;

    initSession((progress) => {
      setModelProgress(Math.round(progress * 100));
    })
      .then((sess) => {
        sessionRef.current = sess;
        setModelReady(true);
      })
      .catch((err) => {
        setError(`Failed to load model: ${err instanceof Error ? err.message : String(err)}`);
        initializingRef.current = false;
      });
  }, [setModelProgress, setModelReady, setError]);

  const runPrompt = useCallback(
    async (text: string) => {
      if (!sessionRef.current) {
        setError("Model not loaded yet");
        return;
      }

      try {
        setError(null);
        setPrompt(text);
        setInferenceRunning(true);

        const ids = encode(text);
        const tokens = tokenize(text);
        setTokens(tokens, ids);

        const { logits, residuals } = await runInference(ids);
        const norms = computeNorms(residuals, ids.length);

        setInferenceResult(logits, residuals, norms);
      } catch (err) {
        setError(
          `Inference failed: ${err instanceof Error ? err.message : String(err)}`,
        );
        setInferenceRunning(false);
      }
    },
    [encode, tokenize, setError, setPrompt, setInferenceRunning, setTokens, setInferenceResult],
  );

  return { runPrompt };
}
