import { encode as gptEncode, decode as gptDecode } from "gpt-tokenizer";

/**
 * Encode text to GPT-2 token IDs.
 */
export function encode(text: string): number[] {
  return gptEncode(text);
}

/**
 * Decode token IDs back to text.
 */
export function decode(ids: number[]): string {
  return gptDecode(ids);
}

/**
 * Tokenize text and return individual token strings for display.
 */
export function tokenize(text: string): string[] {
  const ids = encode(text);
  return ids.map((id) => gptDecode([id]));
}
