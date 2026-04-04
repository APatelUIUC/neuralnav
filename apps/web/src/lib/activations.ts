const HIDDEN_DIM = 768;

/**
 * Compute L2 norms for each (layer, token) position from residual streams.
 * @param residuals - 13 Float32Arrays, each of shape [1, seqLen, 768] flattened
 * @param seqLen - number of tokens in the sequence
 * @returns number[13][seqLen] - L2 norm at each layer/token position
 */
export function computeNorms(
  residuals: Float32Array[],
  seqLen: number,
): number[][] {
  const norms: number[][] = [];

  for (let layer = 0; layer < residuals.length; layer++) {
    const layerNorms: number[] = [];
    const data = residuals[layer]!;

    for (let t = 0; t < seqLen; t++) {
      const offset = t * HIDDEN_DIM;
      let sumSq = 0;
      for (let d = 0; d < HIDDEN_DIM; d++) {
        const val = data[offset + d]!;
        sumSq += val * val;
      }
      layerNorms.push(Math.sqrt(sumSq));
    }

    norms.push(layerNorms);
  }

  return norms;
}

/**
 * Extract a single activation vector (768-dim) for a given layer and token position.
 */
export function extractActivation(
  residuals: Float32Array[],
  layer: number,
  tokenIdx: number,
): Float32Array {
  const data = residuals[layer]!;
  const offset = tokenIdx * HIDDEN_DIM;
  return data.slice(offset, offset + HIDDEN_DIM);
}
