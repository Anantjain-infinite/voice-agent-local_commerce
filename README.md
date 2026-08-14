## Day 9 — Hand Off to a Specialist Agent

**Specialist chosen:** Returns & Refunds Specialist, per the track table. Its job is
narrow on purpose — explain the return policy, check whether a specific item is
eligible, start a return request. It does NOT do general shopping (no catalogue lookup,
no order totals) — if asked, it says so and stays in its lane.

**A bigger change than it looks like:** implementing the handoff surfaced a real bug in
how Days 4–8 stored state. `caller_id` and the Day 8 outcome-tracking flags used to live
as plain attributes on the `Assistant` instance (`self.caller_id`, etc.). A handoff
creates a *new* Agent instance (the specialist) — so those attributes would have been
silently lost the moment a handoff happened, breaking memory and analytics for any call
that used a specialist. Fixed by moving all of that into `session.userdata` (a
`CallState` dataclass), which is shared across every agent in the session — this is
LiveKit's own recommended pattern for exactly this problem. Every tool now takes
`context: RunContext[CallState]` and reads/writes `context.userdata` instead of `self`.
Days 4–8 behavior is unchanged from the outside; only where the state lives changed.

**Shared tools via mixin:** `create_escalation` and `opt_out_of_calls` moved into a
`SharedToolsMixin` that both `Assistant` and `ReturnsSpecialist` inherit from — so a
caller can still ask for a human, or ask to stop being called, no matter which agent is
currently active.

**The handoff itself (Step 3–5):** `transfer_to_returns_specialist` on the main
`Assistant` returns `(ReturnsSpecialist(chat_ctx=self.chat_ctx.copy(...)), message)` —
the tuple-return pattern hands off control, the message is what the caller hears as the
transition, and passing `chat_ctx` means the specialist sees the prior conversation and
doesn't make the caller repeat themselves (Step 4). `ReturnsSpecialist.on_enter()`
introduces itself immediately after taking over (Step 5).

**Boundary with Day 7:** don't confuse this with `create_escalation`. The specialist
handoff is for ordinary return/refund questions the AI can actually answer (policy,
eligibility, starting a return). An actual *dispute* (wrong charge, refund promised but
never received) or an explicit request for a human still goes straight to
`create_escalation` — a specialist AI can't resolve those, only a person can.

**Data:** `returns_policy.py` is a small hand-built policy (7-day window, groceries
non-returnable once opened) — same honesty note as `catalogue.py`, no real policy API
exists to connect to. `returns.py` stores return requests the same way `escalations.py`
stores escalations (`RET-0001` style references).

**To test (Step 6):** ask a normal shopping question first ("what's the price of a
wireless mouse") — should stay with the main agent. Then say "I want to return this" —
should hand off, with the specialist introducing itself and picking up from context.

## Day 8 — Call Analytics Dashboard

**Success definition (Step 1):** per the track table — "the caller finds a product or
completes an enquiry." Operationalized as: during the call, the agent either (a) found
real product matches via `lookup_products`, (b) priced a real order via
`compute_order_total`, or (c) successfully created a human escalation via
`create_escalation` (Day 7) — since handing an unresolvable problem to a human *is*
completing the enquiry, matching how the brief treats escalation as success for Health
Access and Disaster Response. Anything else — caller hangs up with nothing resolved,
catalogue down and nothing else landed — is recorded as failed. **This is a judgment
call, not the only valid one** — if you'd rather only count self-serve resolutions as
success (excluding escalations), that's one line to change in `agent.py`'s
`create_escalation` tool (remove the `self._mark_outcome("escalated")` call).

**How outcomes get recorded (Step 2):** each `Assistant` instance tracks
`outcome_achieved` / `outcome_reason` in memory, set by the three tools above (first
success wins if more than one happens in a call). At call end, a shutdown callback
(`ctx.add_shutdown_callback`, registered once per call in both the inbound and outbound
entrypoint branches) writes exactly one row to a new `calls` table via `call_stats.py` —
same SQLite file as everything else. A call that never actually connects (e.g. outbound
dial rejected, do-not-call skip) is never counted, since no enquiry could have been
completed or failed.

**The dashboard (Step 3–4):** `/calls` on the existing Flask app from Day 7 — three big
numbers (Total / Successful / Failed) pulled live from `call_stats.get_summary()`, plus
a recent-calls table, auto-refreshing every 15s. Nothing is hardcoded; every number
reflects whatever's actually in `bazaar_mitra.db`.

**Privacy (Step 6):** `call_stats.list_recent_calls()` doesn't select `caller_id` from
the database at all — not just hidden in the template, actually excluded from the SQL
query, so it can't leak onto the page by accident. No transcript, name, or contact info
of any kind is ever queried for this view.

**To test (Step 5):** run the agent, have a shopping conversation that actually
resolves (e.g. ask about a real product and get a price), hang up, then refresh
`/calls` — Total and Successful should both go up by one. A conversation where you hang
up without asking about anything real should only increase Total and Failed.


## Day 7 — Know When to Ask for Human Help

**Two triggers, chosen deliberately (not every unresolved question escalates):**
1. Payment, refund, or order dispute.
2. Caller explicitly asks to speak to a human or the shop owner.

Everything else the agent can't verify (e.g. delivery dates) still gets the old
"please contact the shop" line — only these two create an actual ticket.

**Consent is enforced in the prompt, not assumed:** the agent must tell the caller
what it's about to send and get a clear "yes" before calling `create_escalation` —
see the ESCALATION CONSENT RULE in `agent.py`'s system prompt. Say no, and it falls
back to the plain "contact the shop" line instead.

**What gets sent** (never a transcript, never passwords/OTPs/PINs/account numbers):
who needs help (name + caller ID for callback), a short factual summary of what
happened, what the agent already checked, urgency (low/medium/high), the caller's
language, and how they'd like to be followed up with.

**Where it goes:** a local SQLite table (`escalations.py`, same DB file as Day 4's
caller memory) — no external accounts needed. `dashboard.py` is a small local Flask
app that lists open/resolved requests and lets you mark one resolved.

**Reference ID:** every escalation gets one like `ESC-0004`, given to the caller as
their next step, with an honest (not overpromised) description of what happens next.

**To view the dashboard:**
```bash
python dashboard.py
# open http://127.0.0.1:5000
```

**To test both paths (Step 7):** have one call where you say "I want a refund, I was
overcharged" (should trigger an escalation, after asking permission) and one normal
shopping call (should never call `create_escalation`).

## Day 6 — Outbound Calls

**Use case:** order confirmation — the agent calls a customer back about a specific order
to confirm it, on behalf of the shop that took it.

**How it works:** `trigger_outbound_call.py` dispatches the agent with the phone number,
shop, and order summary as job metadata (after checking the local do-not-call list).
Inside `agent.py`, the entrypoint checks for that metadata: if present, it dials the
number via `ctx.api.sip.create_sip_participant(...)` and blocks until the call is
actually picked up (`wait_until_answered=True`) — the agent session is only started
*after* pickup, so nobody hears a greeting play into a still-ringing phone. If there's
no phone number in the metadata, the exact same entrypoint falls through to the normal
inbound (phone or web) flow from Day 4/5, unchanged.

**Opening disclosure (Step 4):** the outbound opening is built by
`build_outbound_opening()` and instructs the agent that, within its first two sentences,
it must say who's calling (Bazaar Mitra, on behalf of the shop), why (confirming this
specific order), and that the caller can ask it to stop at any time. The agent speaks
first on outbound calls — it doesn't wait for the callee, since they didn't ask for the call.

**Opt-out is enforced, not just promised:** if the caller says "stop calling me" at any
point, the agent calls `opt_out_of_calls`, which marks `do_not_call: true` in that
caller's saved facts and ends the call. `trigger_outbound_call.py` checks this flag
before dialing anyone again, so an opt-out actually prevents future calls rather than
just being acknowledged in the moment.

**Setup required before this works** (see `create_outbound_trunk.py` and the setup
notes at the top of it): a Twilio phone number, an Elastic SIP Trunk with a Termination
URI and credential list, and a LiveKit outbound trunk ID (`ST_xxxx`) referencing them,
saved as `LIVEKIT_OUTBOUND_TRUNK_ID` in `.env.local`.

**To place a call:**
```bash
python trigger_outbound_call.py +919876543210 \
    --shop "Sharma Kirana" \
    --order-summary "2kg atta, 1 wireless mouse - Rs. 513"
```

## Day 5 — Catalogue & Order Total Tool

**What it does:** two function tools the agent calls itself —
- `lookup_products(query, shop?)` — real price & stock for a product/category.
- `compute_order_total(items)` — prices out a list of {product, quantity, shop?} using
  real catalogue data, flags anything it couldn't resolve (not found / not enough stock),
  and never silently invents a number.

**Data source:** there's no public, real-time API for neighbourhood/kirana shop inventory
in India, so `catalogue.py` uses a small hand-built local dataset (3 example shops,
~11 products) instead of a live feed. This is called out here per the assignment's
honesty requirement. The dataset is isolated behind one function, `_fetch_catalogue()`
in `catalogue.py` — swapping in a real vendor inventory API or POS integration later
only means rewriting that one function; the tool wiring, timeout handling, and the
"as of" dating around it don't need to change.

**Freshness:** every response includes an `as_of` date (`CATALOGUE_LAST_UPDATED` in
`catalogue.py`). The agent is instructed to mention it when quoting a price or total,
so "yesterday's rate" vs "today's rate" is never ambiguous to the caller.

**Failure handling:** catalogue lookups go through a 3-second timeout. If it's exceeded,
the tool returns `{"ok": False, "error": "..."}` instead of hanging or returning partial
data, and the agent is instructed to say the price list is temporarily unreachable rather
than guess. 

# Voice Agent Starter — Powered by Murf Falcon

Build a production voice AI agent in 5 minutes. Powered by the fastest TTS on the market - swap the system prompt to build anything from customer support to language tutors.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Murf Falcon](https://img.shields.io/badge/TTS-Murf%20Falcon-6366F1)](https://murf.ai/api/docs/text-to-speech/streaming) [![LiveKit](https://img.shields.io/badge/Transport-LiveKit-002cf2)](https://docs.livekit.io) [![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?logo=typescript&logoColor=white)](https://www.typescriptlang.org/) [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

---

## Why Murf Falcon

- **55ms model latency** - fastest production TTS
- **130ms time-to-first-audio** across 10+ global regions
- **$0.01/1000 characters** - up to 10x cheaper than alternatives
- **150+ voices** across 35+ languages
- **99.38% pronunciation accuracy**

---

## Architecture

```mermaid
flowchart LR
    A[🎙️ User speaks] -->|audio| B[Deepgram STT]
    B -->|text| C[LLM]
    C -->|response text| D[Murf Falcon TTS]
    D -->|audio| E[LiveKit]
    E -->|stream| F[🔊 User hears]

    style A fill:#444441,stroke:#888780,color:#fff
    style B fill:#185FA5,stroke:#85B7EB,color:#fff
    style C fill:#534AB7,stroke:#AFA9EC,color:#fff
    style D fill:#0F6E56,stroke:#5DCAA5,color:#fff
    style E fill:#D85A30,stroke:#F0997B,color:#fff
    style F fill:#444441,stroke:#888780,color:#fff
```

---

## Quickstart

### Prerequisites

- **Python** 3.10+
- **[uv](https://docs.astral.sh/uv/)** - fast Python package manager
  ```bash
  # macOS/Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **Node.js** 18+
- **pnpm** — fast Node package manager
  ```bash
  npm install -g pnpm
  ```
- A [LiveKit](https://cloud.livekit.io/) project (free tier available)

### Step 1: Clone the repo

```bash
git clone https://github.com/murf-ai/murf-livekit-starter.git
cd murf-livekit-starter
```

### Step 2: Set up environment variables

Create `.env.local` in both `backend/` and `frontend/` (copy from `.env.example` in each). You need:

| Variable                               | Where to get it                                        | Required |
| -------------------------------------- | ------------------------------------------------------ | -------- |
| `LIVEKIT_URL`                          | LiveKit Cloud dashboard                                | Yes      |
| `LIVEKIT_API_KEY`                      | LiveKit Cloud dashboard                                | Yes      |
| `LIVEKIT_API_SECRET`                   | LiveKit Cloud dashboard                                | Yes      |
| `MURF_API_KEY`                         | [murf.ai/api/dashboard](https://murf.ai/api/dashboard) | Yes      |
| `DEEPGRAM_API_KEY`                     | [deepgram.com](https://deepgram.com)                   | Yes      |
| `GOOGLE_API_KEY` (or `OPENAI_API_KEY`) | Depends on LLM choice                                  | Yes      |

### Step 3: Install backend dependencies

```bash
cd backend
uv sync
uv run python src/agent.py download-files
```

### Step 4: Install frontend dependencies

```bash
cd frontend
pnpm install
```

### Step 5: Run it

**Option A - All-in-one (from repo root):**

```bash
# macOS/Linux
chmod +x start_app.sh
./start_app.sh

# Windows (PowerShell)
.\start_app.ps1
```

**Option B - Separate terminals:**

```bash
# Terminal 1 — LiveKit Server
livekit-server --dev

# Terminal 2 — Backend agent
cd backend && uv run python src/agent.py dev

# Terminal 3 — Frontend
cd frontend && pnpm dev
```

Then open **http://localhost:3000** in your browser.

You should now see the voice agent UI. Click **Start talking**, allow microphone access, and speak — the agent will respond with Murf Falcon TTS. Ensure your backend and (if using Option B) LiveKit server are running.

---

## Deploy

Want to deploy this beyond localhost? You'll need to deploy **two services**: the backend agent and the frontend. Both must use the same LiveKit project.

> This is a two-service app — the backend agent and the frontend UI deploy separately. You'll need both running and connected to the same LiveKit project.

### Backend (Python agent) — Deploy to Railway

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy/tIVCF1?referralCode=cNjn2P&utm_medium=integration&utm_source=template&utm_campaign=generic)

Set these environment variables in Railway:

- `MURF_API_KEY`
- `DEEPGRAM_API_KEY`
- `GOOGLE_API_KEY` or `OPENAI_API_KEY`
- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`

The backend runs as a long-lived Python process that connects to LiveKit as an agent. Railway handles this well.

### Frontend (Next.js) — Deploy to Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/murf-ai/murf-livekit-starter&root-directory=frontend&env=LIVEKIT_URL,LIVEKIT_API_KEY,LIVEKIT_API_SECRET&project-name=murf-voice-agent&repository-name=murf-voice-agent)

Set these environment variables in Vercel:

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `AGENT_NAME` (optional — for explicit agent dispatch)

The frontend is a standard Next.js app. Point it at the same LiveKit instance your backend agent is connected to.

### Connecting them

The frontend and backend don't call each other directly — they both connect to **LiveKit**, which handles the real-time audio transport.

1. Use the **same** `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` on both Railway and Vercel
2. Set `AGENT_NAME=my-agent` on Vercel — this matches the `agent_name="my-agent"` registered in `backend/src/agent.py`
3. Verify: Railway logs should show the agent connected to LiveKit. Open your Vercel URL, click **Start talking** — the agent should respond

If the agent doesn't connect, double-check that both services point to the same LiveKit project and that the backend is running (check Railway logs).

---

## Change the Use Case

The default system prompt makes this a **customer support agent**. You can change the agent’s behavior by editing the prompt.

**Where the prompt lives:** `backend/src/agent.py`- the `SYSTEM_PROMPT` constant (near the top of the file, after the imports). Change that string to change what your voice agent does.

### Example prompts (copy-paste)

**Customer Support (default):**

```
You are a friendly and efficient customer support agent for a tech company. Help users with account issues, billing questions, and product troubleshooting. Be concise, empathetic, and solution-oriented. If you don't know something, say so honestly and offer to escalate.
```

**Language Tutor:**

```
You are a patient and encouraging language tutor helping the user practice conversational Spanish. Speak primarily in Spanish but switch to English to explain grammar or vocabulary when needed. Correct mistakes gently and suggest better phrasing. Keep conversations natural and fun.
```

**AI Receptionist:**

```
You are a professional receptionist for a medical clinic. Help callers schedule appointments, answer questions about office hours and services, and take messages for doctors. Be warm but efficient. Ask for the caller's name and reason for calling upfront.
```

See the Configuration section below for voice, STT, and LLM options.

---

## Configuration

### Murf voice

Edit the `tts=murf.TTS(...)` call in `backend/src/agent.py`. Set the `voice` argument to any Murf voice ID. Examples:

- `Anisha` — Indian English (female, default in this starter)
- `Pooja` — Indian English (female)
- `Samar` — Indian English (male)
- `Amara` — US English (female)
- `Gordon` — US English (male)
- `Hazel` — UK English (female)
- `Bertie` — UK English (male)

Browse all voices: [Murf Voice Library](https://murf.ai/api/docs/voices-styles/voice-library).

### STT provider

STT is configured in `backend/src/agent.py` in the `AgentSession(stt=...)` call. The default is Deepgram (`deepgram.STT(model="nova-3")`). You can swap to another LiveKit-compatible STT plugin if needed.

### LLM (Gemini vs OpenAI)

- **Gemini (default):** Set `GOOGLE_API_KEY` and use `llm=google.LLM(model="gemini-3.5-flash-lite")` in `agent.py`.
- **OpenAI:** Set `OPENAI_API_KEY`, add the OpenAI plugin, and use the corresponding `llm=openai.LLM(...)` in `agent.py`.

### Audio format

Murf Falcon and LiveKit handle audio format internally. For advanced options, see [Murf API docs](https://murf.ai/api/docs) and [LiveKit docs](https://docs.livekit.io).

---

## Project Structure

```
murf-livekit-starter/
├── backend/                 # Python voice agent (LiveKit Agents + Murf Falcon)
│   ├── src/
│   │   └── agent.py         # Agent entrypoint, pipeline (STT/LLM/TTS), system prompt
│   ├── tests/               # Agent tests
│   ├── .env.example         # Backend env template
│   ├── pyproject.toml       # Python deps (uv)
│   └── railway.toml         # Railway deploy config
├── frontend/                # Next.js UI for voice sessions
│   ├── app/
│   │   ├── page.tsx         # Main page
│   │   └── api/token/       # LiveKit token endpoint (dev)
│   ├── components/          # UI (agents-ui, app config, theme)
│   ├── app-config.ts        # Branding, title, button text, accent
│   ├── .env.example         # Frontend env template
│   └── package.json         # Node deps (pnpm)
├── start_app.sh             # Start LiveKit + backend + frontend (macOS/Linux)
├── start_app.ps1            # Start LiveKit + backend + frontend (Windows)
├── README.md                # This file
```

For deeper documentation on each part, see:

- [Backend Documentation](./backend/README.md) — agent pipeline, voice/LLM/STT configuration, testing, deployment
- [Frontend Documentation](./frontend/README.md) — UI customization, visualizers, theming, component architecture

---

## Links

- [Murf API Docs](https://murf.ai/api/docs)
- [Murf Voice Library](https://murf.ai/api/docs/voices-styles/voice-library)
- [LiveKit Docs](https://docs.livekit.io)
- [Deepgram Docs](https://developers.deepgram.com)
- [Murf Falcon Benchmarks](https://murf.ai/falcon/benchmarks)
- [TTS Latency Benchmarker](https://github.com/sahilsgupta/tts-latency-benchmarker) — run your own p50/p95 tests across providers
- [Murf Discord](https://discord.gg/FbKAy96Sz7)
- [Murf Startup Incubator](https://murf.ai/api) — 50M free characters for startups

---

## License

MIT
