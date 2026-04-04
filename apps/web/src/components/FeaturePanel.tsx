import { useStore } from "../store/store";
import type { Feature } from "../store/store";

interface FeaturePanelProps {
  onAblate: (featureIndex: number, featureActivation: number, featureLabel: string) => void;
}

function FeatureRow({
  feature,
  maxActivation,
  onAblate,
}: {
  feature: Feature;
  maxActivation: number;
  onAblate: (featureIndex: number, featureActivation: number, featureLabel: string) => void;
}) {
  const barWidth = maxActivation > 0 ? (feature.activation / maxActivation) * 100 : 0;

  return (
    <div
      className="p-3 rounded-lg bg-[var(--ctp-surface0)] border border-[var(--ctp-surface1)]
        hover:border-[var(--ctp-surface2)] transition-colors duration-150"
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span
            className="px-2 py-0.5 rounded-md text-xs font-mono font-bold
              bg-[var(--ctp-mauve)]/20 text-[var(--ctp-mauve)]"
          >
            #{feature.index}
          </span>
          <span className="text-xs text-[var(--ctp-subtext1)] truncate max-w-[160px]">
            {feature.label}
          </span>
        </div>
        <button
          onClick={() => onAblate(feature.index, feature.activation, feature.label)}
          className="px-2 py-1 rounded-md text-[10px] font-medium
            bg-[var(--ctp-red)]/15 text-[var(--ctp-red)]
            hover:bg-[var(--ctp-red)]/25 transition-colors duration-150"
        >
          Ablate
        </button>
      </div>

      {/* Activation bar */}
      <div className="h-2 rounded-full bg-[var(--ctp-mantle)] overflow-hidden mb-2">
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{
            width: `${barWidth}%`,
            background: "linear-gradient(90deg, var(--ctp-blue), var(--ctp-sapphire))",
          }}
        />
      </div>

      <div className="flex items-center justify-between text-[10px] text-[var(--ctp-overlay0)] mb-1.5">
        <span>activation: {feature.activation.toFixed(3)}</span>
      </div>

      {/* Logit effects */}
      {(feature.top_positive_logits.length > 0 || feature.top_negative_logits.length > 0) && (
        <div className="flex flex-wrap gap-1">
          {feature.top_positive_logits.slice(0, 3).map((entry, i) => (
            <span
              key={`pos-${i}`}
              className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono"
              style={{
                backgroundColor: "rgba(166, 227, 161, 0.15)",
                color: "#a6e3a1",
              }}
            >
              +{entry.logit.toFixed(2)} {entry.token}
            </span>
          ))}
          {feature.top_negative_logits.slice(0, 3).map((entry, i) => (
            <span
              key={`neg-${i}`}
              className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono"
              style={{
                backgroundColor: "rgba(243, 139, 168, 0.15)",
                color: "#f38ba8",
              }}
            >
              {entry.logit.toFixed(2)} {entry.token}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function SkeletonRow() {
  return (
    <div className="p-3 rounded-lg bg-[var(--ctp-surface0)] border border-[var(--ctp-surface1)]">
      <div className="flex items-center gap-2 mb-2">
        <div className="skeleton w-12 h-5" />
        <div className="skeleton w-28 h-4" />
      </div>
      <div className="skeleton w-full h-2 rounded-full mb-2" />
      <div className="flex gap-1">
        <div className="skeleton w-16 h-4" />
        <div className="skeleton w-14 h-4" />
        <div className="skeleton w-18 h-4" />
      </div>
    </div>
  );
}

export default function FeaturePanel({ onAblate }: FeaturePanelProps) {
  const selectedToken = useStore((s) => s.selectedToken);
  const selectedLayer = useStore((s) => s.selectedLayer);
  const tokens = useStore((s) => s.tokens);
  const features = useStore((s) => s.features);
  const featuresLoading = useStore((s) => s.featuresLoading);
  const reconstructionScore = useStore((s) => s.reconstructionScore);

  // Empty state
  if (selectedToken === null || selectedLayer === null) {
    return (
      <div className="h-full flex items-center justify-center text-[var(--ctp-overlay0)] text-sm p-6">
        <div className="text-center">
          <svg
            className="mx-auto w-10 h-10 mb-3 opacity-40"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122"
            />
          </svg>
          <p>Click a cell in the heatmap</p>
          <p className="text-xs mt-1 text-[var(--ctp-overlay0)]/60">
            to see SAE feature decomposition
          </p>
        </div>
      </div>
    );
  }

  const tokenText = tokens[selectedToken] ?? "";
  const displayFeatures = features?.slice(0, 20) ?? [];
  const maxActivation = displayFeatures.length > 0
    ? Math.max(...displayFeatures.map((f) => f.activation))
    : 1;

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[var(--ctp-surface1)]">
        <h2 className="text-sm font-semibold text-[var(--ctp-text)]">
          Layer {selectedLayer}{" "}
          <span className="text-[var(--ctp-overlay1)]">&middot;</span>{" "}
          <span className="text-[var(--ctp-lavender)]">
            &ldquo;{tokenText}&rdquo;
          </span>
        </h2>
        {features && (
          <p className="text-xs text-[var(--ctp-subtext0)] mt-1">
            {features.length} features active
            {reconstructionScore !== null && (
              <>
                {" "}
                &middot; reconstruction:{" "}
                <span className="text-[var(--ctp-green)] font-mono">
                  {(reconstructionScore * 100).toFixed(1)}%
                </span>
              </>
            )}
          </p>
        )}
      </div>

      {/* Feature list */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {featuresLoading ? (
          Array.from({ length: 6 }).map((_, i) => <SkeletonRow key={i} />)
        ) : displayFeatures.length > 0 ? (
          displayFeatures.map((feature) => (
            <FeatureRow
              key={feature.index}
              feature={feature}
              maxActivation={maxActivation}
              onAblate={onAblate}
            />
          ))
        ) : (
          <div className="text-center text-xs text-[var(--ctp-overlay0)] py-8">
            No features returned from server.
          </div>
        )}
      </div>
    </div>
  );
}
