// generateObject on Crusoe: a typed object validated against a Zod schema. Incurs inference charges.
import { generateObject } from "ai";
import { z } from "zod";

import { crusoe, DEFAULT_MODEL } from "./crusoe.js";

const siteAssessment = z.object({
  site: z.string(),
  verdict: z.enum(["GO", "CONDITIONAL", "NO_GO"]),
  score: z.number().int().min(0).max(100),
  risks: z.array(z.string()).max(3),
});

const { object } = await generateObject({
  model: crusoe(DEFAULT_MODEL),
  schema: siteAssessment,
  system: "You assess sites for a 100 MW AI data center. Be decisive.",
  prompt:
    "Narvik, Norway: hydro power at 2.5 cents per kWh, 4 C average temperature, " +
    "fiber to Oslo and London, one 420 kV substation with spare capacity.",
});

console.log(JSON.stringify(object, null, 2));
