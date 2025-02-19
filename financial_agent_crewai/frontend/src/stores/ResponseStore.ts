import { create } from "zustand";

interface ResponseState {
  response: string;
  setResponse: (response: string) => void;
}

export const useResponseStore = create<ResponseState>((set) => ({
  response: "",
  setResponse: (response) => set({ response }),
}));
