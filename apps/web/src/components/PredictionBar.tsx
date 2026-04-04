import { useMemo } from "react";
import { useStore } from "../store/store";
import { decode } from "../lib/tokenizer";

function topKSoftmax(logits: Float32Array, topK: number): { token: string; prob: number }[] {
  const indices: number[] = [];
  for (let i = 0; i < logits.length; i++) indices.push(i);
  indices.sort((a, b) => logits[b]! - logits[a]!);
  const topIndices = indices.slice(0, topK);

  const maxVal = logits[topIndices[0]!]!;
  const vals: number[] = [];
  for (const idx of topIndices) vals.push(Math.exp(logits[idx]! - maxVal));

  let fullSum = 0;
  for (let i = 0; i < logits.length; i++) fullSum += Math.exp(logits[i]! - maxVal);

  return topIndices.map((idx, i) => ({
    token: decode([idx]),
    prob: vals[i]! / fullSum,
  }));
}

function TokenChips({
  predictions,
  highlight,
}: {
  predictions: { token: string; prob: number }[];
  highlight: "mauve" | "peach";
}) {
  const colors = highlight === "mauve"
    ? { activeBg: "rgba(203, 166, 247, 0.2)", activeText: "var(--ctp-mauve)" }
    : { activeBg: "rgba(250, 179, 135, 0.2)", activeText: "var(--ctp-peach)" };

  return (
    <div className="flex items-center gap-1.5 overflow-x-auto">
      {predictions.map((pred, i) => (
        <span
          key={i}
          className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-mono whitespace-nowrap"
          style={{
            backgroundColor: i === 0 ? colors.activeBg : "rgba(108, 112, 134, 0.15)",
            color: i === 0 ? colors.activeText : "var(--ctp-subtext0)",
          }}
        >
          <span className="font-semibold">{pred.token || "\\n"}</span>
          <span style={{ opacity: 0.6 }}>{(pred.prob * 100).toFixed(1)}%</span>
        </span>
      ))}
    </div>
  );
}

export default function PredictionBar() {
  const logits = useStore((s) => s.logits);
  const tokens = useStore((s) => s.tokens);
  const ablation = useStore((s) => s.ablation);

  const vocabSize = 50257;

  const lastLogits = useMemo(() => {
    if (!logits || tokens.length === 0) return null;
    const offset = (tokens.length - 1) * vocabSize;
    return logits.slice(offset, offset + vocabSize);
  }, [logits, tokens]);

  const predictions = useMemo(() => {
    if (!lastLogits) return null;
    return topKSoftmax(lastLogits, 8);
  }, [lastLogits]);

  const ablatedPredictions = useMemo(() => {
    if (!lastLogits || !ablation?.logitDelta) return null;
    // Apply logit delta to original logits
    const modified = new Float32Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) {
      modified[i] = lastLogits[i]! + ablation.logitDelta[i]!;
    }
    return topKSoftmax(modified, 8);
  }, [lastLogits, ablation]);

  if (!predictions) return null;

  return (
    <div className="border-b border-[var(--ctp-surface1)] bg-[var(--ctp-mantle)]">
      <div className="px-6 py-2 flex items-center gap-3">
        <span className="text-[11px] text-[var(--ctp-overlay1)] font-medium whitespace-nowrap">
          Next token
        </span>
        <TokenChips predictions={predictions} highlight="mauve" />
      </div>

      {ablatedPredictions && ablation && (
        <div className="px-6 py-2 border-t border-[var(--ctp-surface0)] flex items-center gap-3">
          <span className="text-[11px] text-[var(--ctp-peach)] font-medium whitespace-nowrap">
            Without #{ablation.featureIndex}
          </span>
          <TokenChips predictions={ablatedPredictions} highlight="peach" />
        </div>
      )}
    </div>
  );
}
