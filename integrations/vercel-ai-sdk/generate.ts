// generateText on Crusoe with a system prompt and token usage. Incurs inference charges.
import { generateText } from "ai";

import { crusoe, DEFAULT_MODEL } from "./crusoe.js";

const { text, usage } = await generateText({
  model: crusoe(DEFAULT_MODEL),
  system: "You are a concise infrastructure engineer. Answer in two sentences.",
  prompt: "Why does serverless inference suit a product that has not found traffic yet?",
});

console.log(text);
console.log("\n[usage]", usage);
