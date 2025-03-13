export type JsonValueType =
  | string
  | number
  | boolean
  | null
  | JsonObjectType
  | JsonArrayType;
export type JsonObjectType = { [key: string]: JsonValueType };
export type JsonArrayType = JsonValueType[];
export type DropdownOptionType = {
  id: string;
  label: string;
  disabled?: boolean;
  disabled_reason?: string;
};
