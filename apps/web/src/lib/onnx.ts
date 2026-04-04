import * as ort from "onnxruntime-web";

const DEFAULT_MODEL_URL =
  "https://huggingface.co/clueless-wanderer/gpt2-onnx-residuals/resolve/main/gpt2_with_residuals.onnx";

let session: ort.InferenceSession | null = null;

/**
 * Download the ONNX model with progress tracking via ReadableStream.
 */
async function downloadModel(
  onProgress: (fraction: number) => void,
): Promise<ArrayBuffer> {
  const url = import.meta.env.VITE_MODEL_URL || DEFAULT_MODEL_URL;

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download model: ${response.status}`);
  }

  const contentLength = response.headers.get("content-length");
  const total = contentLength ? parseInt(contentLength, 10) : 0;

  if (!response.body) {
    // Fallback if ReadableStream not supported
    const buffer = await response.arrayBuffer();
    onProgress(1);
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total > 0) {
      onProgress(received / total);
    }
  }

  const result = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }

  onProgress(1);
  return result.buffer;
}

/**
 * Initialize the ONNX InferenceSession, downloading the model if needed.
 */
export async function initSession(
  onProgress: (fraction: number) => void,
): Promise<ort.InferenceSession> {
  if (session) return session;

  // Use WASM backend
  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

  const modelBuffer = await downloadModel(onProgress);

  session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ["wasm"],
  });

  return session;
}

/**
 * Run inference on token IDs.
 * Returns logits and 13 residual stream arrays (layers 0-12).
 */
export async function runInference(inputIds: number[]): Promise<{
  logits: Float32Array;
  residuals: Float32Array[];
}> {
  if (!session) {
    throw new Error("Model session not initialized. Call initSession first.");
  }

  const seqLen = inputIds.length;

  // ONNX Runtime expects BigInt64Array for int64 inputs
  const inputData = new BigInt64Array(inputIds.map((id) => BigInt(id)));
  const inputTensor = new ort.Tensor("int64", inputData, [1, seqLen]);

  const feeds: Record<string, ort.Tensor> = { input_ids: inputTensor };
  const results = await session.run(feeds);

  // Extract logits
  const logitsOutput = results["logits"];
  if (!logitsOutput) {
    throw new Error("Model output missing 'logits'");
  }
  const logits = new Float32Array(logitsOutput.data as Float32Array);

  // Extract residual streams (resid_0 through resid_12)
  const residuals: Float32Array[] = [];
  for (let i = 0; i <= 12; i++) {
    const key = `resid_${i}`;
    const resid = results[key];
    if (!resid) {
      throw new Error(`Model output missing '${key}'`);
    }
    residuals.push(new Float32Array(resid.data as Float32Array));
  }

  return { logits, residuals };
}

/**
 * Check if the model session is loaded.
 */
export function isSessionReady(): boolean {
  return session !== null;
}
