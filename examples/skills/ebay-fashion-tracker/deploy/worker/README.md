# eBay deletion-notice responder

eBay keeps a production keyset disabled until it is subscribed to [marketplace account deletion notifications](https://developer.ebay.com/marketplace-account-deletion). The endpoint must be public HTTPS, so it cannot run on your laptop. This Cloudflare Worker is the smallest thing that satisfies it: it answers eBay's challenge and acknowledges notices.

It only acknowledges because the tracker stores no eBay user identifiers. If you change the tracker to store seller names, this responder has to delete them when a notice arrives.

## Deploy

You need a free Cloudflare account and Node.js.

```bash
cd deploy/worker

# 1. Choose a verification token: 32 to 80 characters, letters, digits, _ and -.
python3 -c "import secrets; print(secrets.token_urlsafe(48))"

# 2. Store it as a secret, then deploy.
npx wrangler secret put VERIFICATION_TOKEN
npx wrangler deploy
```

`wrangler deploy` prints the Worker's URL. Put that exact URL in `ENDPOINT_URL` in `wrangler.toml` and run `npx wrangler deploy` again. The challenge hash includes the URL, so it has to match character for character what you give eBay.

## Register it with eBay

In the developer portal, open **Application Keys > Production > Notifications** (the "Alerts & Notifications" page):

1. Enter an alert email address and save.
2. Enter the Worker URL as the notification endpoint and the same verification token, and save. eBay sends the challenge immediately; saving succeeds only if the Worker answers correctly.
3. Use **Send Test Notification** to confirm a POST is acknowledged.

Your production keyset is enabled once the subscription is saved. Check it with `eft doctor ebay`.

## Remove

```bash
npx wrangler delete
```
