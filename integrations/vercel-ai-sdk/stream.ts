// streamText on Crusoe, written to stdout as tokens arrive. Incurs inference charges.
import { streamText } from "ai";

import { crusoe } from "./crusoe.js";

const result = streamText({
  model: crusoe("deepseek-ai/DeepSeek-V4-Pro"),
  prompt: "Write a haiku about sustainable computing, then explain it in one sentence.",
});

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
process.stdout.write("\n");
