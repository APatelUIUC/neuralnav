import { useState } from "react";
import { useStore } from "../store/store";

const EXAMPLE_PROMPTS = [
  "The Eiffel Tower is in",
  "When John and Mary went to the store, John gave the bag to",
  "1 + 1 =",
  "The capital of France is",
  "def fibonacci(n):",
];

interface PromptInputProps {
  onSubmit: (text: string) => void;
}

export default function PromptInput({ onSubmit }: PromptInputProps) {
  const [text, setText] = useState("The capital of France is");
  const inferenceRunning = useStore((s) => s.inferenceRunning);
  const modelReady = useStore((s) => s.modelReady);

  const handleSubmit = () => {
    const trimmed = text.trim();
    if (trimmed && !inferenceRunning && modelReady) {
      onSubmit(trimmed);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="w-full px-6 py-4">
      <div className="flex gap-3 items-center">
        <div className="relative flex-1">
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="The capital of France is"
            disabled={!modelReady || inferenceRunning}
            className="w-full px-4 py-3 rounded-xl text-sm
              bg-[var(--ctp-surface0)] text-[var(--ctp-text)]
              border border-[var(--ctp-surface2)]
              placeholder:text-[var(--ctp-overlay0)]
              focus:outline-none focus:border-[var(--ctp-lavender)]
              focus:ring-1 focus:ring-[var(--ctp-lavender)]
              disabled:opacity-50 transition-all duration-200"
          />
        </div>
        <button
          onClick={handleSubmit}
          disabled={!modelReady || inferenceRunning || !text.trim()}
          className="px-6 py-3 rounded-xl text-sm font-medium
            bg-[var(--ctp-mauve)] text-[var(--ctp-crust)]
            hover:brightness-110 active:brightness-90
            disabled:opacity-40 disabled:cursor-not-allowed
            transition-all duration-200 whitespace-nowrap"
        >
          {inferenceRunning ? (
            <span className="flex items-center gap-2">
              <svg
                className="animate-spin h-4 w-4"
                viewBox="0 0 24 24"
                fill="none"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              Running...
            </span>
          ) : (
            "Run Inference"
          )}
        </button>
      </div>

      <div className="flex flex-wrap gap-2 mt-3">
        {EXAMPLE_PROMPTS.map((prompt) => (
          <button
            key={prompt}
            onClick={() => {
              setText(prompt);
              if (modelReady && !inferenceRunning) {
                onSubmit(prompt);
              }
            }}
            disabled={!modelReady || inferenceRunning}
            className="px-3 py-1.5 rounded-lg text-xs
              bg-[var(--ctp-surface0)] text-[var(--ctp-subtext0)]
              border border-[var(--ctp-surface1)]
              hover:border-[var(--ctp-lavender)] hover:text-[var(--ctp-text)]
              disabled:opacity-40 disabled:cursor-not-allowed
              transition-all duration-200 cursor-pointer"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}
