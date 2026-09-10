# Passive history ingress v1

Authenticated clients can attach to an existing conversation, read a bounded text snapshot,
and save finalized speech or other external dialogue. These operations never create an agent,
run tools, submit work, mint approvals, create a session, or reopen a closed session.

Discover support before sending anything. The gateway advertises `features.passive_history`
in `GET /v1/capabilities`; dashboard `/api/status` advertises `passive_history`. Both also
serve the authenticated `GET <prefix>/capabilities`. Require `version: 1` and `passive_only: true`.
An unsupported host must not be replaced with a call to chat merely to persist text.

Gateway prefix: `/v1/passive-history`, using the existing `Authorization: Bearer ...` key.
Named gateway profiles use `/p/{profile}/v1/passive-history` and that profile's key.
Dashboard prefix: `/api/passive-history`, using the existing dashboard session-token header
or verified OAuth identity. Dashboard operations accept the established `?profile=name` selector.
Profile authority and DB resolution remain with the host. Paths and credentials cannot appear in bodies.

| Method | Suffix | Required JSON fields |
|---|---|---|
| POST | `/attach` | `tab_id`, `session_id` |
| POST | `/snapshot` | `tab_id`, `attachment_id`, `generation`, `session_id` |
| POST | `/commit` | attachment fields plus `event_id`, `origin_turn_id`, `messages` |
| POST | `/reconcile` | `session_id`, `event_id` |
| POST | `/detach` | attachment fields |

Unknown fields are rejected. Event, origin, tab and attachment IDs are 1–128 ASCII letters,
digits, dots, underscores or hyphens; use independently generated UUIDs for events/tabs.
Session IDs are exact host-issued strings, at most 256 characters. Generation is a positive integer.

Attach returns the identity fields, canonical profile and a snapshot with `conversation_id`,
current `session_id`, `messages`, `truncated`, and capabilities. Preserve the outer `session_id`
as the original selected target; a compression successor inside the snapshot is not a target switch.
Snapshots contain at most 20 recent user/assistant text rows and 32 KiB combined UTF-8 text.
They omit system prompts, tools, configuration, API sidecars and non-text payloads.
Compaction handoffs use the canonical session display projection: internal summaries are removed,
while genuine earlier text inside a merged carrier remains visible. Internal notification kinds
and empty text rows are omitted; multibyte truncation never fabricates empty messages.

Example commit, using identity fields returned by attach:

```json
{
  "tab_id": "tab-uuid", "attachment_id": "returned-id", "generation": 1,
  "session_id": "original-session-id", "event_id": "event-uuid",
  "origin_turn_id": "utterance-uuid",
  "messages": [{"role": "user", "content": "Finalized spoken text."}]
}
```

Messages must be one finalized `user` or `assistant` message, or an ordered `[user, assistant]`
pair. Only `role` and `content` are accepted. Text is nonempty and at most 64 KiB UTF-8 per
message; the complete HTTP body is capped at 160 KiB. Host-owned provenance is never permission.

`saved` and `already_saved` return a receipt with the original message IDs, conversation/segment
IDs and external-history revision. Receipt IDs are not execution or approval IDs. Equal retries
retain the stable event and origin IDs; changing payload or owner under an existing event conflicts.
Canonical history remains in original user/assistant roles and is visible on the next authorized turn.

Each principal/profile/tab has independent attachment authority. Reattaching replaces only that
tab's attachment. Commit validates the exact original target, principal, profile, attachment ID,
generation and host epoch inside the same SQLite writer transaction as the canonical insertion.
Detach revokes only the exact current generation; an old detach cannot revoke a newer attachment.
An active ordinary turn causes a retryable busy response; passive operations never hold its lease.

A host restart invalidates attachment authority. After a lost response, call read-only reconcile
with the original target and event: `saved` proves the existing receipt, `unknown` proves only that
no receipt was found, and `retired` means its canonical rows were removed. If unknown, reattach to
the original target before retrying a fresh commit. Never retarget a pending event to the newly
selected conversation. Receipt reconciliation works independently of attachments or host epochs.

Session deletion cascades attachment removal. Message deletion invalidates attachments referencing
the affected original, conversation-root or snapshot segment. Receipts retain content-free
tombstones; delete/recreate cannot revive them. Clients discard derived context when attachment
validation fails and must reattach. There is no lifecycle push stream in this version.

Errors have `{error, retryable}`: authentication 401/403; invalid message/body 400; body-size cap 413;
missing target 404; stale attachment, target unavailable, event conflict or busy 409; retired 410;
SQLite unavailable 503. Only busy/store-unavailable are marked retryable. Error bodies omit
transcript contents, paths and credentials. Closed or ambiguous lineages refuse without selecting a sibling.

`origin_adoption` is explicitly **false**. Do not pass an utterance here while an authoritative
execution submission may also persist it. Run acceptance alone is not proof of canonical input;
verified execution-origin adoption requires a separate capability and is not implemented by v1.

Compatibility: schema 32 adds revocable attachment storage over the existing passive receipt
contract. Database recovery deliberately drops attachment authority while retaining receipts.
Rollback to a prior executable leaves additive tables harmless, but clients must treat missing
capabilities as unsupported. Local tests and commits do not establish deployed availability.
