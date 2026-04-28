"""
file_upload.py — Helpers for posting text and uploading files to Slack threads.

All functions accept a Slack WebClient (or the Bolt client object, which
exposes the same interface) so callers don't need to manage the client.
"""

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Text posting
# ─────────────────────────────────────────────────────────────────────────────

def post_text(client, channel_id: str, thread_ts: str, text: str) -> dict:
    """
    Post a plain-text (or Block Kit) message into a Slack thread.

    Args:
        client:      Slack WebClient or Bolt client.
        channel_id:  Target channel.
        thread_ts:   Thread timestamp — reply is posted in this thread.
        text:        Message body (Markdown supported by Slack's mrkdwn).

    Returns:
        Slack API response dict.
    """
    try:
        response = client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=text,
            mrkdwn=True,
        )
        logger.info(f"Posted message to {channel_id} thread {thread_ts}")
        return response
    except Exception as exc:
        logger.error(f"Failed to post text to Slack: {exc}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# File / image upload
# ─────────────────────────────────────────────────────────────────────────────

def upload_file(
    client,
    channel_id: str,
    thread_ts: str,
    file_path: str,
    title: str = "",
    comment: str = "",
) -> dict:
    """
    Upload a single file (e.g. a PNG plot) into a Slack thread.

    Args:
        client:      Slack WebClient or Bolt client.
        channel_id:  Target channel.
        thread_ts:   Thread to attach the file to.
        file_path:   Absolute path to the file on disk.
        title:       Display title shown in Slack (defaults to filename).
        comment:     Optional text caption posted alongside the file.

    Returns:
        Slack API response dict (includes file ID in response["file"]["id"]).
    """
    path = Path(file_path)
    if not path.exists():
        logger.warning(f"upload_file: path does not exist — {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    display_title = title or path.name

    try:
        with open(path, "rb") as fh:
            response = client.files_upload_v2(
                channel=channel_id,
                thread_ts=thread_ts,
                file=fh,
                filename=path.name,
                title=display_title,
                initial_comment=comment or None,
            )
        logger.info(f"Uploaded file '{display_title}' to {channel_id} thread {thread_ts}")
        return response
    except Exception as exc:
        logger.error(f"Failed to upload file '{file_path}': {exc}")
        raise


def upload_multiple_files(
    client,
    channel_id: str,
    thread_ts: str,
    file_paths: List[str],
    captions: List[str] | None = None,
) -> List[dict]:
    """
    Upload multiple files to the same Slack thread, one at a time.

    Args:
        client:      Slack WebClient or Bolt client.
        channel_id:  Target channel.
        thread_ts:   Thread to attach files to.
        file_paths:  List of absolute file paths.
        captions:    Optional list of captions, one per file.

    Returns:
        List of Slack API response dicts (one per file).
    """
    if captions is None:
        captions = [""] * len(file_paths)

    if len(captions) != len(file_paths):
        raise ValueError("captions list length must match file_paths list length.")

    responses = []
    for file_path, caption in zip(file_paths, captions):
        try:
            resp = upload_file(
                client=client,
                channel_id=channel_id,
                thread_ts=thread_ts,
                file_path=file_path,
                comment=caption,
            )
            responses.append(resp)
        except Exception as exc:
            logger.warning(f"Skipping '{file_path}' due to upload error: {exc}")

    return responses
