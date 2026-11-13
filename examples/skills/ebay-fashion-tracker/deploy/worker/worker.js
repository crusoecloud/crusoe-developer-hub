// Responder for eBay Marketplace Account Deletion notifications.
//
// eBay keeps a production keyset disabled until it is subscribed to these notices
// (or the developer declares that the application stores no eBay data). eBay first
// sends a GET with ?challenge_code=...; the reply must be the hex SHA-256 of
// challengeCode + verificationToken + endpointURL, in that order. After that it
// sends POSTs when an eBay user asks for their data to be deleted.
//
// ebay-fashion-tracker stores no eBay user identifiers (no seller or buyer names),
// so there is nothing to delete and a notice only needs to be acknowledged.
//
// Configuration:
//   ENDPOINT_URL        the exact URL you register with eBay (wrangler.toml [vars])
//   VERIFICATION_TOKEN  32 to 80 characters: letters, digits, _ and - (wrangler secret)

function configured(env) {
  const token = env.VERIFICATION_TOKEN || "";
  const endpoint = env.ENDPOINT_URL || "";
  return /^[A-Za-z0-9_-]{32,80}$/.test(token) && endpoint.startsWith("https://") && !endpoint.includes("YOUR-SUBDOMAIN");
}

export default {
  async fetch(request, env) {
    if (!configured(env)) {
      // Answering the challenge with a hash of placeholder values would look like a
      // working endpoint to a browser but fail at eBay. Say what is missing instead.
      return new Response(
        "not configured: set the VERIFICATION_TOKEN secret (32 to 80 characters) and ENDPOINT_URL in wrangler.toml, then deploy again",
        { status: 500 },
      );
    }

    if (request.method === "GET") {
      const code = new URL(request.url).searchParams.get("challenge_code");
      if (!code) {
        return new Response("ok", { status: 200 });
      }
      const input = new TextEncoder().encode(code + env.VERIFICATION_TOKEN + env.ENDPOINT_URL);
      const digest = await crypto.subtle.digest("SHA-256", input);
      const challengeResponse = Array.from(new Uint8Array(digest))
        .map((byte) => byte.toString(16).padStart(2, "0"))
        .join("");
      return new Response(JSON.stringify({ challengeResponse }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }

    if (request.method === "POST") {
      // Nothing to delete: acknowledge so eBay does not retry.
      return new Response(null, { status: 204 });
    }

    return new Response("method not allowed", { status: 405 });
  },
};
