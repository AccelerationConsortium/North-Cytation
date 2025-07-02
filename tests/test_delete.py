import requests

SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/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"

payload = {"text": "🔧 Webhook test: is this thing on?"}
response = requests.post(SLACK_WEBHOOK_URL, json=payload)

if response.status_code == 200:
    print("✅ Webhook is working!")
else:
    print(f"❌ Webhook failed: {response.status_code} – {response.text}")