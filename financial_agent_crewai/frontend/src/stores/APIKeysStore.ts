import { create } from "zustand";

export type APIKeysType = { SambaNova: string | null; Serper: string | null };

interface APIKeysState {
  apiKeys: { SambaNova: string | null; Serper: string | null };
  addApiKey: (apiKeyName: string, apiKeyValue: string) => void;
  updateApiKey: (apiKeyName: string, apiKeyValue: string | null) => void;
}

export const useAPIKeysStore = create<APIKeysState>((set, get) => ({
  apiKeys: { SambaNova: null, Serper: null },
  addApiKey: (apiKeyName, apiKeyValue) =>
    set({ apiKeys: { ...get().apiKeys, [apiKeyName]: apiKeyValue } }),
  updateApiKey: (apiKeyName: string, apiKeyValue: string | null) =>
    set({ apiKeys: { ...get().apiKeys, [apiKeyName]: apiKeyValue } }),
}));
