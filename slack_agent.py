import requests
import logging
import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from requests.exceptions import ConnectionError, Timeout, RequestException
import urllib3.exceptions

# Replace with your Slack webhook URL
def load_secrets(path="secrets.txt"):
    secrets = {}
    with open(path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                secrets[key] = value
    return secrets
secrets = load_secrets()

SLACK_BOT_TOKEN = secrets["SLACK_BOT_TOKEN"]
SLACK_WEBHOOK_URL = secrets["SLACK_WEBHOOK_URL"]
SLACK_CHANNEL = "C08C38E2GEM" 

# ✅ Setup logging
logging.basicConfig(level=logging.INFO)

# ✅ Initialize Slack WebClient
client = WebClient(token=SLACK_BOT_TOKEN)

def test_slack_connectivity():
    """Test if Slack is reachable before attempting to send messages."""
    try:
        # Simple connectivity test with short timeout
        response = requests.get("https://hooks.slack.com", timeout=5)
        return True
    except:
        return False

def send_slack_message(message):
    """Send a message to a Slack channel with robust error handling."""
    try:
        payload = {"text": message}
        response = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)

        if response.status_code == 200:
            print("Message sent successfully!")
            return True
        else:
            print(f"Slack Error: {response.status_code}, {response.text}")
            logging.error(f"Slack HTTP Error: {response.status_code}, {response.text}")
            return False
            
    except (ConnectionError, urllib3.exceptions.NameResolutionError) as e:
        print(f"Slack network error (continuing workflow): {e}")
        logging.error(f"Slack network connectivity issue: {e}")
        return False
        
    except Timeout as e:
        print(f"Slack timeout error (continuing workflow): {e}")
        logging.error(f"Slack request timeout: {e}")
        return False
        
    except RequestException as e:
        print(f"Slack request error (continuing workflow): {e}")
        logging.error(f"Slack request failed: {e}")
        return False
        
    except Exception as e:
        print(f"Unexpected Slack error (continuing workflow): {e}")
        logging.error(f"Unexpected Slack error: {e}")
        return False

def upload_and_post_file(file_path, message):
    """Upload and post file to Slack with robust error handling."""
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
            return True
        else:
            logging.error(f"❌ Failed to post file: {response['error']}")
            return False

    except SlackApiError as e:
        logging.error(f"❌ Slack API Error: {e.response['error']}")
        print(f"Slack API error (continuing workflow): {e.response['error']}")
        return False
        
    except (ConnectionError, urllib3.exceptions.NameResolutionError) as e:
        print(f"Slack network error during file upload (continuing workflow): {e}")
        logging.error(f"Slack network connectivity issue during file upload: {e}")
        return False
        
    except Exception as e:
        print(f"Unexpected Slack file upload error (continuing workflow): {e}")
        logging.error(f"Unexpected Slack file upload error: {e}")
        return False

def safe_send_slack_message(message, silent_fail=True):
    """
    Safe wrapper for send_slack_message that never crashes the workflow.
    
    Args:
        message (str): Message to send
        silent_fail (bool): If True, only log errors. If False, print errors.
    
    Returns:
        bool: True if message sent successfully, False otherwise
    """
    try:
        return send_slack_message(message)
    except Exception as e:
        error_msg = f"Critical Slack error (workflow continuing): {e}"
        logging.error(error_msg)
        if not silent_fail:
            print(error_msg)
        return False
