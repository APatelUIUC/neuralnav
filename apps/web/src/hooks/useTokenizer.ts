import { useCallback } from "react";
import { encode, decode, tokenize } from "../lib/tokenizer";

export function useTokenizer() {
  const encodeText = useCallback((text: string) => encode(text), []);
  const decodeIds = useCallback((ids: number[]) => decode(ids), []);
  const tokenizeText = useCallback((text: string) => tokenize(text), []);

  return { encode: encodeText, decode: decodeIds, tokenize: tokenizeText };
}
