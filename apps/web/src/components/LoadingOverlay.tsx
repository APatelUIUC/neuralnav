import { useStore } from "../store/store";

export default function LoadingOverlay() {
  const modelReady = useStore((s) => s.modelReady);
  const modelProgress = useStore((s) => s.modelProgress);
  const error = useStore((s) => s.error);

  if (modelReady) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[var(--ctp-crust)]/95 backdrop-blur-sm">
      <div className="text-center max-w-md px-8">
        {/* Logo / icon */}
        <div className="mb-6">
          <svg
            className="mx-auto w-16 h-16 text-[var(--ctp-mauve)]"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5"
            />
          </svg>
        </div>

        <h1 className="text-xl font-bold text-[var(--ctp-text)] mb-2">
          SAE Explorer
        </h1>

        <p className="text-sm text-[var(--ctp-subtext0)] mb-6">
          {error
            ? error
            : modelProgress < 100
              ? "Downloading GPT-2 Small (~120MB)..."
              : "Initializing inference session..."}
        </p>

        {!error && (
          <div className="w-full">
            {/* Progress bar */}
            <div className="h-3 rounded-full bg-[var(--ctp-surface0)] overflow-hidden">
              <div
                className="h-full rounded-full progress-bar-fill transition-all duration-300 ease-out"
                style={{
                  width: `${modelProgress}%`,
                  background:
                    "linear-gradient(90deg, var(--ctp-mauve), var(--ctp-pink))",
                }}
              />
            </div>
            <p className="text-xs text-[var(--ctp-overlay0)] mt-2">
              {modelProgress}%
            </p>
          </div>
        )}

        {error && (
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 rounded-lg text-sm font-medium
              bg-[var(--ctp-surface0)] text-[var(--ctp-text)]
              hover:bg-[var(--ctp-surface1)] transition-colors"
          >
            Retry
          </button>
        )}
      </div>
    </div>
  );
}
