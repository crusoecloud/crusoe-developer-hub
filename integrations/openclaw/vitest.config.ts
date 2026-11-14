import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

function testSdk(name: string): string {
  return fileURLToPath(new URL(`./test/sdk/${name}.ts`, import.meta.url));
}

export default defineConfig({
  resolve: {
    alias: {
      "openclaw/plugin-sdk/plugin-entry": testSdk("plugin-entry"),
      "openclaw/plugin-sdk/provider-auth": testSdk("provider-auth"),
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
