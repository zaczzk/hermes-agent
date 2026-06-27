"""Telegram setMessageReaction tool for the agent.

Lets the agent set emoji reactions on the user's most recent Telegram
message mid-conversation. Used to surface the full lifecycle chain
(👀 auto → 🔎 → 🛠️ → ⏳ → 📝 → ✅/👍) when working on long requests,
without spamming the chat with status messages.

Reads three env vars at call time:
- TELEGRAM_BOT_TOKEN: the bot's HTTP API token (set at gateway startup)
- HERMES_LAST_USER_CHAT_ID: chat id of the user whose message we're reacting to
  (set by the gateway per agent run, in gateway/run.py:_run_agent)
- HERMES_LAST_USER_MESSAGE_ID: message id of that user's triggering message
  (same source)

If the env vars are missing the tool returns a structured error rather than
crashing — useful in CLI mode where there's no Telegram session.
"""

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict

logger = logging.getLogger(__name__)


# Lifecycle chain — recommended for "long" agent runs (5+ tool calls):
#   🔎  reviewing/reading
#   🛠️  actively working (mid-tool)
#   ⏳  this is taking a while
#   📝  finalising (writing the response)
#   ✅  done (the gateway auto-puts 👍 on success; use ✅ to override
#       the 👍 with a custom final reaction)
# The gateway auto-puts 👀 on receive and 👍/👎 on lifecycle end —
# this tool only needs to fill the mid-flight phases.
_LIFECYCLE_EMOJIS = ("🔎", "🛠️", "⏳", "📝", "✅")
_DEFAULT_EMOJI = "🛠️"


def _call_set_message_reaction(
    bot_token: str, chat_id: str, message_id: str, emoji: str
) -> Dict[str, Any]:
    """POST setMessageReaction via the Telegram Bot HTTP API.

    Returns the parsed JSON body on success, or a structured error dict.
    """
    url = f"https://api.telegram.org/bot{bot_token}/setMessageReaction"
    # Telegram expects `reaction` as a JSON array of ReactionType (we use emoji).
    payload = {"chat_id": chat_id, "message_id": message_id, "reaction": emoji}
    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # Telegram returns 4xx with a JSON body describing the error
        try:
            err_body = exc.read().decode("utf-8")
        except Exception:  # pragma: no cover - defensive
            err_body = ""
        return {
            "ok": False,
            "error": f"HTTP {exc.code} {exc.reason}",
            "body": err_body,
        }
    except urllib.error.URLError as exc:
        return {"ok": False, "error": f"URL error: {exc.reason}"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "error": f"unexpected: {exc!r}"}

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return {"ok": False, "error": "non-JSON response", "body": body[:500]}

    return {
        "ok": bool(parsed.get("ok")),
        "description": parsed.get("description"),
        "result": parsed.get("result"),
    }


def _set_reaction(emoji: str = _DEFAULT_EMOJI) -> str:
    """Handler for the `set_reaction` tool.

    Sets a single emoji reaction on the user's most recent Telegram message.
    The chat_id and message_id come from env vars that the gateway sets
    per session (see gateway/run.py:_run_agent). The bot token comes from
    TELEGRAM_BOT_TOKEN.

    Args:
        emoji: a single emoji to react with. Defaults to 🛠️. Common
            choices: 🔎 (reviewing), 🛠️ (working), ⏳ (long task),
            📝 (finalising), ✅ (done), 👍 (success), 👎 (failure).

    Returns:
        JSON string with `ok`, `emoji`, `chat_id`, `message_id` keys
        on success, or `ok=false` and `error` on failure.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("HERMES_LAST_USER_CHAT_ID", "")
    message_id = os.getenv("HERMES_LAST_USER_MESSAGE_ID", "")

    if not bot_token:
        return json.dumps({
            "ok": False,
            "error": "TELEGRAM_BOT_TOKEN not set in env (not a Telegram session?)",
        })
    if not chat_id or not message_id:
        return json.dumps({
            "ok": False,
            "error": "HERMES_LAST_USER_CHAT_ID / HERMES_LAST_USER_MESSAGE_ID not set "
                     "(gateway did not bind a session for this agent run)",
        })

    result = _call_set_message_reaction(bot_token, chat_id, message_id, emoji)
    result["emoji"] = emoji
    result["chat_id"] = chat_id
    result["message_id"] = message_id

    if not result["ok"]:
        logger.warning(
            "set_reaction failed (emoji=%s chat_id=%s msg_id=%s): %s",
            emoji, chat_id, message_id, result.get("error") or result.get("description"),
        )

    return json.dumps(result)


def _check_reaction_available() -> bool:
    """Tool is only available when both the bot token and session context exist.

    Without the session context vars, the agent isn't in a Telegram session
    (e.g. CLI) and there's nothing to react to. Hiding the tool in that
    case keeps the schema lean.
    """
    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        return False
    if not (os.getenv("HERMES_LAST_USER_CHAT_ID") and os.getenv("HERMES_LAST_USER_MESSAGE_ID")):
        return False
    return True


SET_REACTION_SCHEMA = {
    "name": "set_reaction",
    "description": (
        "Set a single emoji reaction on the user's most recent Telegram message "
        "to surface the agent's lifecycle state. The gateway already auto-puts "
        "👀 on receive and 👍/👎 on success/failure, so this tool is for the "
        "MID-FLIGHT phases only. Recommended chain for long tasks: 🔎 (reviewing) "
        "→ 🛠️ (working on it, mid-tool) → ⏳ (this is taking a while, 5+ tool "
        "calls) → 📝 (finalising response) → ✅ (override the auto-👍 to mark "
        "custom completion). Skip this tool entirely for short acks — the "
        "lifecycle reactions are enough."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "emoji": {
                "type": "string",
                "description": (
                    "Single emoji to react with. Recommended: 🔎, 🛠️, ⏳, 📝, ✅. "
                    "Default: 🛠️."
                ),
                "default": _DEFAULT_EMOJI,
            },
        },
        "required": [],
    },
}


# Registration — follows the homeassistant_tool.py pattern.
from tools.registry import registry  # noqa: E402  (import after definitions per pattern)

registry.register(
    name="set_reaction",
    toolset="set_reaction",
    schema=SET_REACTION_SCHEMA,
    handler=_set_reaction,
    check_fn=_check_reaction_available,
    emoji="🎯",
)
