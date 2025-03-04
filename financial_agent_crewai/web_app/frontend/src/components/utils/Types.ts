export type JsonValueType =
  | string
  | number
  | boolean
  | null
  | JsonObjectType
  | JsonArrayType;
export type JsonObjectType = { [key: string]: JsonValueType };
export type JsonArrayType = JsonValueType[];
