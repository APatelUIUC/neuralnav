import { useStore } from "./store/store";
import { useGPT2 } from "./hooks/useGPT2";
import { useSAE } from "./hooks/useSAE";
import LoadingOverlay from "./components/LoadingOverlay";
import PromptInput from "./components/PromptInput";
import Heatmap from "./components/Heatmap";
import FeaturePanel from "./components/FeaturePanel";
import AblationDiff from "./components/AblationDiff";

export default function App() {
  const { runPrompt } = useGPT2();
  const { runAblation, resetAblation } = useSAE();
  const error = useStore((s) => s.error);
  const modelReady = useStore((s) => s.modelReady);

  return (
    <div className="h-screen flex flex-col bg-[var(--ctp-base)] overflow-hidden">
      <LoadingOverlay />

      {/* Error banner */}
      {error && modelReady && (
        <div className="px-6 py-2 bg-[var(--ctp-red)]/15 border-b border-[var(--ctp-red)]/30">
          <p className="text-xs text-[var(--ctp-red)]">{error}</p>
        </div>
      )}

      {/* Top bar: logo + prompt input */}
      <header className="border-b border-[var(--ctp-surface1)] bg-[var(--ctp-mantle)]">
        <div className="flex items-center gap-3 px-6 pt-4 pb-0">
          <div className="flex items-center gap-2">
            <svg
              className="w-5 h-5 text-[var(--ctp-mauve)]"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5"
              />
            </svg>
            <h1 className="text-sm font-bold text-[var(--ctp-text)] tracking-tight">
              SAE Explorer
            </h1>
          </div>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-[var(--ctp-mauve)]/15 text-[var(--ctp-mauve)] font-medium">
            GPT-2 Small
          </span>
        </div>
        <PromptInput onSubmit={runPrompt} />
      </header>

      {/* Main content */}
      <main className="flex-1 flex overflow-hidden">
        {/* Heatmap area (left ~60%) */}
        <div className="flex-[3] flex flex-col min-w-0 border-r border-[var(--ctp-surface1)]">
          <Heatmap />
        </div>

        {/* Feature panel (right ~40%) */}
        <div className="flex-[2] min-w-[320px] max-w-[480px] bg-[var(--ctp-base)]">
          <FeaturePanel onAblate={runAblation} />
        </div>
      </main>

      {/* Ablation diff (bottom) */}
      <AblationDiff onReset={resetAblation} />
    </div>
  );
}
