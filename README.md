# Bazaar Mitra — AI Voice Shopping Assistant for Local Commerce

A voice agent for India's local kirana/neighbourhood shops — built over the
[10 Days of Voice Agents — VoiceForBharat Edition](https://github.com/murf-ai/voice-for-bharat-challenge-2026)
challenge, on the [Murf LiveKit Starter](https://github.com/murf-ai/murf-livekit-starter).

Talk to it (by phone or browser) and it can look up real prices and stock, remember you
on your next call, place outbound calls of its own, hand a genuine dispute to a human
instead of pretending it can resolve one, and hand off returns questions to a
specialist agent mid-conversation.

For the full build story — what broke, what I learned, why each feature exists — see
the [companion blog post](https://dev.to/aj_infinite_2208/i-built-a-voice-shopping-assistant-for-local-kirana-stores-in-10-days-heres-everything-that-broke-k50). This README is the practical
"how do I run this" reference.

## What it can do

- **Understand Hindi, English, or code-mixed speech** and reply in kind.
- **Remember returning callers** — name, language, shopping facts — but only after
  explicitly asking permission and getting a "yes."
- **Look up real prices and stock** and price out multi-item orders, never a guessed
  number; says so plainly if the catalogue is temporarily unreachable instead of
  inventing an answer.
- **Make outbound calls** (e.g. order confirmation), always opening with who's calling,
  why, and how to make it stop — and actually honors "stop calling me."
- **Escalate to a human** for the two situations an AI genuinely shouldn't handle alone
  — a payment/order dispute, or an explicit request for a person — with the caller's
  consent, a short factual summary (never a full transcript, never sensitive data), and
  a reference ID.
- **Hand off to a Returns & Refunds specialist** for return-policy questions, without
  making the caller repeat what they already said.
- **Track whether each call actually succeeded** on a small live analytics dashboard —
  no hardcoded numbers, no caller-identifying data shown.

## Architecture

```
Caller (phone or browser)
        │  real-time audio
        ▼
   LiveKit Room  ───────────────────────────────►  Agent Session
                                                      STT → LLM → TTS
                                                   (Deepgram → Gemini → Murf Falcon)
                                                            │
                                          ┌─────────────────┴─────────────────┐
                                          ▼                                   ▼
                                  Function tools                        spoken reply
                             (memory, catalogue,                      streams back to
                              escalation, returns)                     the caller
                                          │
                                          ▼
                                       SQLite
                              (callers · catalogue · escalations
                               · call outcomes · returns)
                                          │
                                          ▼
                                 Human dashboard
                              (open escalations + call
                                  analytics, Flask)
```

A rendered version of this diagram is at `blog/architecture.png`.

State that needs to survive a specialist handoff (caller ID, whether the call has
succeeded yet) lives in `session.userdata`, not on any single agent instance — see
`agent.py`'s `CallState` dataclass. This matters: a handoff creates a brand-new agent
object, so anything stored as `self.something` on the old agent would simply vanish.

## Project structure

```
backend/src/
├── agent.py                   Main entrypoint. Assistant + ReturnsSpecialist agents,
│                               every tool, inbound/outbound call handling.
├── db.py                      Caller memory (SQLite) — name, language, shopping facts.
├── catalogue.py                Product catalogue + order-total tool. Hand-built dataset,
│                               documented as such — see the file's own docstring.
├── escalations.py             Human escalation requests (Day 7).
├── returns.py                 Return requests initiated by the specialist (Day 9).
├── returns_policy.py          Hand-built return/refund policy the specialist checks against.
├── call_stats.py              Call outcome tracking for the analytics dashboard (Day 8).
├── dashboard.py                Flask app: /  (open escalations), /calls (analytics).
├── env_utils.py                Shared helper: finds .env.local reliably regardless of
│                               which directory a script is run from.
├── create_outbound_trunk.py   One-time script: registers your Twilio trunk with LiveKit.
└── trigger_outbound_call.py   Places an outbound call (checks do-not-call list first).

frontend/app/api/connection-details/route.ts
                                Issues the LiveKit token for web callers; assigns a
                                stable per-browser identity via a cookie, so the same
                                browser is recognized across separate calls.
```

## Prerequisites

- Python 3.10+
- Node.js (for the frontend, if you're using the web client)
- A [LiveKit Cloud](https://cloud.livekit.io) project (or self-hosted LiveKit server)
- API keys for [Deepgram](https://deepgram.com) (STT), [Google AI Studio](https://aistudio.google.com) (Gemini), and [Murf](https://murf.ai) (Falcon TTS)
- A [Twilio](https://twilio.com) account — only needed for outbound calling (Day 6);
  everything else works without it

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # macOS/Linux
python -m pip install -r requirements.txt
```

Create `backend/.env.local` (this file is git-ignored — never commit it):

```bash
# Core — required
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
DEEPGRAM_API_KEY=...
GOOGLE_API_KEY=...
MURF_API_KEY=...
AGENT_NAME=my-agent

# Outbound calling — optional, only needed for Day 6 (see below)
LIVEKIT_OUTBOUND_TRUNK_ID=          # filled in by create_outbound_trunk.py
TWILIO_TERMINATION_URI=
TWILIO_PHONE_NUMBER=
SIP_AUTH_USERNAME=
SIP_AUTH_PASSWORD=

# Testing — optional
CATALOGUE_SIMULATE_DOWN=false       # set true to rehearse the catalogue-outage path
```

If you're running the frontend too:
```bash
cd frontend
npm install
```
It reads the same `LIVEKIT_URL`/`LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET`, plus
`AGENT_NAME` if you want explicit agent dispatch, from its own `.env.local`.

## Running it

```bash
cd backend/src
python agent.py dev
```

This starts a worker that waits for a room to join — it does nothing on its own until
a caller connects. Connect either via:
- **The frontend** (`npm run dev` in `frontend/`, then open the local URL), or
- **[LiveKit's Agents Playground](https://agents-playground.livekit.io)** pointed at your project.

Say something ordinary first ("what's the price of a wireless mouse?"), then try the
things that make this more than a Q&A bot: ask it to remember you, ask about a return,
say "I want to talk to a human." The gaps in a voice agent show up in conversation, not
in a unit test.

## The dashboard

```bash
cd backend/src
pip install flask       # if you don't already have it
python dashboard.py
```
Open `http://127.0.0.1:5000` — `/` shows open human-escalation requests, `/calls` shows
live call analytics (total / successful / failed, auto-refreshing). Neither page shows
a transcript, password, or caller-identifying data; `/calls` doesn't even query for it.

## Outbound calling setup (optional)

Only needed if you want the agent to place calls itself (Day 6). Skip this section
entirely if you're just testing inbound/web conversations.

1. Buy a Twilio phone number and create an Elastic SIP Trunk with a Termination URI and
   a credential list (username/password) — see Twilio's Elastic SIP Trunking docs.
2. Add the four `TWILIO_*`/`SIP_*` variables above to `.env.local`.
3. Register the trunk with LiveKit:
   ```bash
   python create_outbound_trunk.py
   ```
   Copy the printed `ST_xxxxxxxx` into `.env.local` as `LIVEKIT_OUTBOUND_TRUNK_ID`.
4. Place a call:
   ```bash
   python trigger_outbound_call.py +91XXXXXXXXXX \
       --shop "Sharma Kirana" \
       --order-summary "2kg atta, 1 wireless mouse - Rs. 513"
   ```

## Known limitations

Being upfront about what's real and what isn't, per this project's own rule about never
inventing data:

- **The product catalogue (`catalogue.py`) and return policy (`returns_policy.py`) are
  hand-built, not connected to a real inventory/POS system.** There's no public API for
  neighbourhood-shop inventory in India, so this is a small illustrative dataset. Both
  are isolated behind single functions specifically so a real integration is a small
  swap, not a rewrite.
- **The dashboard has no authentication.** Fine for local/demo use; needs auth before
  it's anything else.
- **Caller identity for web callers is a browser cookie**, not a verified identity — the
  same person on a different browser looks like a new caller unless they identify
  themselves by phone (the `identify_caller` tool handles that case).
- **The analytics "success" definition is a judgment call**, documented in `agent.py`'s
  `CallState` — a call counts as successful if the caller found a product, priced an
  order, got a return started, or got escalated to a human. Reasonable people could
  define this differently for their own use case.

## What's next

- Real inventory/POS integration in place of the hand-built catalogue.
- Auth on the dashboard.
- More outbound triggers (a restock nudge based on a caller's own order history is
  already sitting in the memory data).
- Real usage data on the dashboard, and a check on whether "success" as defined here
  actually matches what a human reviewing the calls would call a good conversation.

## Credits

Built on the [Murf LiveKit Starter](https://github.com/murf-ai/murf-livekit-starter),
using [Murf Falcon](https://murf.ai) for text-to-speech, as part of
[10 Days of Voice Agents — VoiceForBharat Edition](https://github.com/murf-ai/voice-for-bharat-challenge-2026).