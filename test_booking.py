from pathlib import Path
from email.message import EmailMessage
from datetime import datetime
import smtplib
import ssl


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
    Times are interpreted as Toronto local time
    by the Power Automate flow.
    """

    sender = secrets["GMAIL_ADDRESS"]
    password = secrets["GMAIL_APP_PASSWORD"].replace(" ", "")
    recipient = secrets["BOOKING_RECIPIENT"]

    # Validate dates before sending
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    if start_dt >= end_dt:
        raise ValueError("End time must be after start time.")

    if "\n" in title or "\r" in title:
        raise ValueError("Title must be a single line.")

    # Create the booking email
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = "NORTH_CYTATION_BOOKING"

    # Format expected by Power Automate
    msg.set_content(
    f"NCBOOK|{title}|{start}|{end}|ENDNC"
)

    # Connect to Gmail using STARTTLS on port 587
    context = ssl.create_default_context()

    try:
        with smtplib.SMTP(
            "smtp.gmail.com", 587, timeout=30
        ) as smtp:

            smtp.ehlo()
            smtp.starttls(context=context)
            smtp.ehlo()

            # Show supported authentication methods (not passwords)
            print("SMTP authentication methods:", smtp.esmtp_features.get("auth"))

            smtp.login(sender, password)
            smtp.send_message(msg)

    except smtplib.SMTPAuthenticationError as error:
        print("Gmail authentication failed.")
        print("Check the Gmail address and app password.")
        raise

    except (smtplib.SMTPException, OSError) as error:
        print(f"SMTP connection or sending failed: {error}")
        raise

    print("\nBooking request email sent successfully.")
    print(f"Title: {title}")
    print(f"Start: {start}")
    print(f"End:   {end}")
    print("Check North Cytation in Outlook to verify the booking.")


# --------------------------------------------------
# Test booking
# --------------------------------------------------

if __name__ == "__main__":
    book_robot(
        title="Python Test Booking",
        start="2026-11-03T15:00:00",
        end="2026-11-03T15:05:00"
    )
