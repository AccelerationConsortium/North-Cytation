
from pathlib import Path
from email.message import EmailMessage
import smtplib


# --------------------------------------------------
# Load credentials from secrets.txt
# --------------------------------------------------

def load_secrets(filename="secrets.txt"):
    path = Path(__file__).resolve().parent / filename

    secrets = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        key, value = line.split("=", 1)
        secrets[key.strip()] = value.strip()

    return secrets


secrets = load_secrets()


# --------------------------------------------------
# Book North Cytation
# --------------------------------------------------

def book_robot(title, start, end):
    """
    Send a booking request to Power Automate.

    Date format: YYYY-MM-DDTHH:MM:SS
    Times are in Toronto local time.
    """

    sender = secrets["GMAIL_ADDRESS"]
    password = secrets["GMAIL_APP_PASSWORD"].replace(" ", "")
    recipient = secrets["BOOKING_RECIPIENT"]

    # Must match Power Automate subject filter
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = "NORTH_CYTATION_BOOKING"

    # Must match the 3-line format Power Automate expects
    msg.set_content(f"{title}\n{start}\n{end}")

    # Send through Gmail
    with smtplib.SMTP_SSL(
        "smtp.gmail.com", 465, timeout=30
    ) as smtp:
        smtp.login(sender, password)
        smtp.send_message(msg)

    print("Booking request sent successfully.")
    print(f"Title: {title}")
    print(f"Start: {start}")
    print(f"End:   {end}")
    print("Check North Cytation in Outlook to confirm the event.")


# --------------------------------------------------
# Test booking
# --------------------------------------------------

if __name__ == "__main__":
    book_robot(
        title="Python Test Booking",
        start="2026-11-03T15:00:00",
        end="2026-11-03T15:05:00"
    )
