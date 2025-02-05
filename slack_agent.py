import requests
import logging
import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Replace with your Slack webhook URL
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/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"
SLACK_BOT_TOKEN = "xoxb-1089129130001-8411300097889-QinoradzIThH3GKQ0cmbE6Jl"
SLACK_CHANNEL = "C08C38E2GEM" 

# ✅ Setup logging
logging.basicConfig(level=logging.INFO)

# ✅ Initialize Slack WebClient
client = WebClient(token=SLACK_BOT_TOKEN)

def send_slack_message(message):
    """Send a message to a Slack channel."""
    payload = {"text": message}
    response = requests.post(SLACK_WEBHOOK_URL, json=payload)

    if response.status_code == 200:
        print("Message sent successfully!")
    else:
        print(f"Error: {response.status_code}, {response.text}")

def upload_and_post_file(file_path, message):
    try:
        # ✅ Step 1: Upload File
        logging.info(f"Uploading file: {file_path}")
        new_file = client.files_upload_v2(
            title=os.path.basename(file_path),
            filename=os.path.basename(file_path),
            file=open(file_path, "rb"),
        )
        
        file_url = new_file["file"]["permalink"]  # Get public link
        logging.info(f"✅ File uploaded successfully: {file_url}")

        # ✅ Step 2: Post Message with File Link
        response = client.chat_postMessage(
            channel=SLACK_CHANNEL,
            text=message+f"{file_url}",
        )

        if response["ok"]:
            logging.info(f"✅ File posted successfully in {SLACK_CHANNEL}")
        else:
            logging.error(f"❌ Failed to post file: {response['error']}")

    except SlackApiError as e:
        logging.error(f"❌ Slack API Error: {e.response['error']}")
