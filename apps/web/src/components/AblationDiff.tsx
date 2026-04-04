import { useStore } from "../store/store";

interface AblationDiffProps {
  onReset: () => void;
}

function TokenRow({
  token,
  value,
  maxAbs,
  type,
}: {
  token: string;
  value: number;
  maxAbs: number;
  type: "promoted" | "demoted";
}) {
  const barPct = maxAbs > 0 ? (Math.abs(value) / maxAbs) * 100 : 0;
  const barColor = type === "promoted" ? "var(--ctp-green)" : "var(--ctp-red)";

  return (
    <div className="flex items-center gap-2">
      <span className="text-[11px] font-mono text-[var(--ctp-text)] w-20 truncate text-right">
        {token}
      </span>
      <div className="flex-1 h-4 rounded bg-[var(--ctp-mantle)] overflow-hidden">
        <div
          className="h-full rounded transition-all duration-300"
          style={{
            width: `${barPct}%`,
            backgroundColor: barColor,
            opacity: 0.8,
          }}
        />
      </div>
      <span className="text-[10px] font-mono text-[var(--ctp-overlay1)] w-14 text-right">
        {value > 0 ? "+" : ""}{value.toFixed(3)}
      </span>
    </div>
  );
}

export default function AblationDiff({ onReset }: AblationDiffProps) {
  const ablation = useStore((s) => s.ablation);
  const ablationLoading = useStore((s) => s.ablationLoading);

  if (!ablation && !ablationLoading) return null;

  const allAbs = [
    ...(ablation?.tokensPromoted.map((t) => Math.abs(t.logitChange)) ?? []),
    ...(ablation?.tokensDemoted.map((t) => Math.abs(t.logitChange)) ?? []),
  ];
  const maxAbs = allAbs.length > 0 ? Math.max(...allAbs) : 1;

  return (
    <div className="border-t border-[var(--ctp-surface1)] bg-[var(--ctp-mantle)]">
      <div className="px-6 py-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-sm font-semibold text-[var(--ctp-text)]">
              Ablation Result
              {ablation && (
                <span className="text-[var(--ctp-peach)] ml-2">
                  Feature #{ablation.featureIndex}
                  {ablation.featureLabel && ` — ${ablation.featureLabel}`}
                </span>
              )}
            </h3>
            <p className="text-[10px] text-[var(--ctp-overlay0)] mt-0.5">
              Approximate logit lens projection. Shows tokens most affected by removing this feature.
            </p>
          </div>
          <button
            onClick={onReset}
            className="px-3 py-1.5 rounded-lg text-xs font-medium
              bg-[var(--ctp-surface0)] text-[var(--ctp-subtext1)]
              hover:bg-[var(--ctp-surface1)] transition-colors duration-150"
          >
            Reset
          </button>
        </div>

        {ablationLoading ? (
          <div className="flex gap-8">
            <div className="flex-1 space-y-2">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="skeleton h-4 rounded" />
              ))}
            </div>
            <div className="flex-1 space-y-2">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="skeleton h-4 rounded" />
              ))}
            </div>
          </div>
        ) : ablation ? (
          <div className="flex gap-8">
            <div className="flex-1 min-w-0">
              <h4 className="text-xs font-semibold text-[var(--ctp-green)] mb-2 uppercase tracking-wider">
                Tokens Promoted
              </h4>
              <div className="space-y-1.5">
                {ablation.tokensPromoted.map((item, i) => (
                  <TokenRow
                    key={i}
                    token={item.token}
                    value={item.logitChange}
                    maxAbs={maxAbs}
                    type="promoted"
                  />
                ))}
              </div>
            </div>
            <div className="flex-1 min-w-0">
              <h4 className="text-xs font-semibold text-[var(--ctp-red)] mb-2 uppercase tracking-wider">
                Tokens Demoted
              </h4>
              <div className="space-y-1.5">
                {ablation.tokensDemoted.map((item, i) => (
                  <TokenRow
                    key={i}
                    token={item.token}
                    value={item.logitChange}
                    maxAbs={maxAbs}
                    type="demoted"
                  />
                ))}
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
