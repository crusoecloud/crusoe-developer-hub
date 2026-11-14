import type { ProviderAuthMethod } from "openclaw/plugin-sdk/plugin-entry";
import type { ProviderApiKeyAuthMethodOptions } from "openclaw/plugin-sdk/provider-auth";

export function createProviderApiKeyAuthMethod(
  params: ProviderApiKeyAuthMethodOptions,
): ProviderAuthMethod {
  return {
    id: params.methodId,
    label: params.label,
    hint: params.hint,
    kind: "api_key",
    starterModel: params.defaultModel,
    run: async () => {
      throw new Error("The test SDK does not store credentials.");
    },
  };
}
