import type {
  DefinePluginEntryOptions,
  DefinedPluginEntry,
} from "openclaw/plugin-sdk/plugin-entry";

export function definePluginEntry({
  id,
  name,
  description,
  register,
}: DefinePluginEntryOptions): DefinedPluginEntry {
  return { id, name, description, register };
}
