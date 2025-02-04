import requests

class SlackMessenger:

    # Replace with your Slack webhook URL
    SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/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"

    def send_slack_message(self,message):
        """Send a message to a Slack channel."""
        payload = {"text": message}
        response = requests.post(self.SLACK_WEBHOOK_URL, json=payload)

        if response.status_code == 200:
            print("Message sent successfully!")
        else:
            print(f"Error: {response.status_code}, {response.text}")

    