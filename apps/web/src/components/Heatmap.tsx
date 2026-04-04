import { useCallback, useEffect, useRef, useState } from "react";
import { useStore } from "../store/store";
import { normToRGB, colorLegendStops } from "../lib/colors";

const CELL_W = 48;
const CELL_H = 36;
const LABEL_LEFT = 60;
const LABEL_TOP = 80;
const LEGEND_WIDTH = 24;
const LEGEND_GAP = 16;

interface TooltipState {
  x: number;
  y: number;
  norm: number;
  token: string;
  layer: number;
}

export default function Heatmap() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  const norms = useStore((s) => s.norms);
  const tokens = useStore((s) => s.tokens);
  const selectedToken = useStore((s) => s.selectedToken);
  const selectedLayer = useStore((s) => s.selectedLayer);
  const selectCell = useStore((s) => s.selectCell);

  const numLayers = norms ? norms.length : 0;
  const seqLen = tokens.length;

  // Find global min/max for color scale
  const { min: globalMin, max: globalMax } = (() => {
    if (!norms) return { min: 0, max: 1 };
    let min = Infinity;
    let max = -Infinity;
    for (const layerNorms of norms) {
      for (const v of layerNorms) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    return { min, max };
  })();

  const canvasWidth = LABEL_LEFT + seqLen * CELL_W + LEGEND_GAP + LEGEND_WIDTH + 40;
  const canvasHeight = LABEL_TOP + numLayers * CELL_H + 20;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !norms) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasWidth * dpr;
    canvas.height = canvasHeight * dpr;
    canvas.style.width = `${canvasWidth}px`;
    canvas.style.height = `${canvasHeight}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // Draw token labels (top)
    ctx.font = "11px 'Fira Code', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";

    for (let t = 0; t < seqLen; t++) {
      const x = LABEL_LEFT + t * CELL_W + CELL_W / 2;
      const tokenStr = tokens[t] ?? "";
      // Truncate long tokens
      const display = tokenStr.length > 6 ? tokenStr.slice(0, 5) + "\u2026" : tokenStr;

      ctx.save();
      ctx.translate(x, LABEL_TOP - 8);
      ctx.rotate(-Math.PI / 4);
      ctx.fillStyle =
        selectedToken === t ? "#b4befe" : "#a6adc8";
      ctx.fillText(display, 0, 0);
      ctx.restore();
    }

    // Draw layer labels (left)
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    for (let l = 0; l < numLayers; l++) {
      const y = LABEL_TOP + l * CELL_H + CELL_H / 2;
      ctx.fillStyle =
        selectedLayer === l ? "#b4befe" : "#a6adc8";
      ctx.font = "11px 'Fira Code', monospace";
      ctx.fillText(`L${l}`, LABEL_LEFT - 10, y);
    }

    // Draw cells
    for (let l = 0; l < numLayers; l++) {
      for (let t = 0; t < seqLen; t++) {
        const norm = norms[l]?.[t] ?? 0;
        const [r, g, b] = normToRGB(norm, globalMin, globalMax);
        const x = LABEL_LEFT + t * CELL_W;
        const y = LABEL_TOP + l * CELL_H;

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.beginPath();
        ctx.roundRect(x + 1, y + 1, CELL_W - 2, CELL_H - 2, 4);
        ctx.fill();

        // Selected cell highlight
        if (selectedToken === t && selectedLayer === l) {
          ctx.strokeStyle = "#f5c2e7";
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.roundRect(x, y, CELL_W, CELL_H, 5);
          ctx.stroke();
        }
      }
    }

    // Draw color legend
    const legendX = LABEL_LEFT + seqLen * CELL_W + LEGEND_GAP;
    const legendY = LABEL_TOP;
    const legendH = numLayers * CELL_H;

    const stops = colorLegendStops(globalMin, globalMax, 50);
    const stripH = legendH / stops.length;

    for (let i = 0; i < stops.length; i++) {
      ctx.fillStyle = stops[stops.length - 1 - i]!.color;
      ctx.fillRect(legendX, legendY + i * stripH, LEGEND_WIDTH, stripH + 1);
    }

    // Legend border
    ctx.strokeStyle = "#45475a";
    ctx.lineWidth = 1;
    ctx.strokeRect(legendX, legendY, LEGEND_WIDTH, legendH);

    // Legend labels
    ctx.fillStyle = "#a6adc8";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(globalMax.toFixed(1), legendX + LEGEND_WIDTH + 4, legendY - 2);
    ctx.textBaseline = "bottom";
    ctx.fillText(globalMin.toFixed(1), legendX + LEGEND_WIDTH + 4, legendY + legendH + 2);
  }, [norms, tokens, seqLen, numLayers, selectedToken, selectedLayer, globalMin, globalMax, canvasWidth, canvasHeight]);

  useEffect(() => {
    draw();
  }, [draw]);

  const getCellFromEvent = (
    e: React.MouseEvent<HTMLCanvasElement>,
  ): { tokenIdx: number; layerIdx: number } | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const col = Math.floor((mx - LABEL_LEFT) / CELL_W);
    const row = Math.floor((my - LABEL_TOP) / CELL_H);

    if (col >= 0 && col < seqLen && row >= 0 && row < numLayers) {
      return { tokenIdx: col, layerIdx: row };
    }
    return null;
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const cell = getCellFromEvent(e);
    if (cell) {
      selectCell(cell.tokenIdx, cell.layerIdx);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const cell = getCellFromEvent(e);
    if (cell && norms) {
      const norm = norms[cell.layerIdx]?.[cell.tokenIdx] ?? 0;
      setTooltip({
        x: e.clientX + 12,
        y: e.clientY - 12,
        norm,
        token: tokens[cell.tokenIdx] ?? "",
        layer: cell.layerIdx,
      });
    } else {
      setTooltip(null);
    }
  };

  const handleMouseLeave = () => {
    setTooltip(null);
  };

  if (!norms || seqLen === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-[var(--ctp-overlay0)] text-sm">
        <div className="text-center">
          <div className="text-4xl mb-4 opacity-30">
            <svg
              className="mx-auto w-16 h-16"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1}
                d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2"
              />
            </svg>
          </div>
          <p>Enter a prompt and run inference to see the activation heatmap</p>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="flex-1 overflow-auto p-4">
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className="cursor-crosshair"
        style={{ minWidth: canvasWidth, minHeight: canvasHeight }}
      />
      {tooltip && (
        <div
          className="heatmap-tooltip"
          style={{ left: tooltip.x, top: tooltip.y }}
        >
          <div className="font-mono font-bold">
            L{tooltip.layer} &middot;{" "}
            <span className="text-[var(--ctp-lavender)]">
              &ldquo;{tooltip.token}&rdquo;
            </span>
          </div>
          <div className="text-[var(--ctp-subtext0)]">
            norm: {tooltip.norm.toFixed(4)}
          </div>
        </div>
      )}
    </div>
  );
}
