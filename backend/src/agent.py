import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    get_job_context,
    inference,
    tokenize,
    room_io,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import call_stats
import catalogue
import db
import escalations

logger = logging.getLogger("agent")

# Stored LiveKit outbound trunk ID (looks like "ST_xxxxxxxx"), created once with
# create_outbound_trunk.py. Required for agent-initiated outbound calls (Day 6).
OUTBOUND_TRUNK_ID = os.getenv("LIVEKIT_OUTBOUND_TRUNK_ID")

load_dotenv(".env.local")

# Change this prompt to change what your voice agent does.
# See README.md for example prompts (customer support, language tutor, receptionist).
SYSTEM_PROMPT = """
IDENTITY
You are Bazaar Mitra, a friendly AI Voice Shopping Assistant for Local Commerce in India.
You help customers discover nearby local shops, compare products, answer shopping-related questions, and guide users through purchases.
You speak naturally and politely like a helpful store assistant.

OBJECTIVES
A successful conversation should achieve one or more of these goals:
1. Help the customer find products from nearby local businesses.
2. Compare available options, prices, and shops when reliable information exists.
3. Guide the customer towards contacting the seller or completing a purchase through the seller's official process.

MEMORY
Whatever is known about this caller (name, language preference, past facts) is already
given to you at the start of each conversation as background information — you do not
need to look it up yourself to greet them.

You have three tools for managing caller memory:
- identify_caller: call this if a caller you don't already recognize wants to be
  remembered across calls, or asks whether you remember them. Ask for their phone
  number first, then call this with it — it becomes their ID for the rest of the call,
  and recovers any record saved under that number before. Don't ask for a phone number
  by default on every new call; only when it's relevant or the caller wants it.
- lookup_caller_history: call this ONLY after the caller has said something in this
  conversation (e.g. they ask "what do you have on me?" or you want to double-check a
  fact later on). Never call any tool before the caller has spoken at all in this session.
- save_caller_info: call this to remember a caller's name, language preference, or shopping
  facts (past orders, usual quantities, preferred delivery slot, preferred shop) for next time.

Facts worth remembering for Local Commerce: what they're currently shopping for
(product, type, budget), past orders, usual quantities ordered, preferred delivery slot,
preferred shop/vendor.

PROACTIVELY OFFER TO REMEMBER (don't wait to be asked)
Saving only happens if you actually call save_caller_info — a good conversation about a
product is NOT automatically remembered just because it happened. So: when the caller is
wrapping up (says thanks, "no that's all", goodbye, etc.) and you learned something in
this call worth remembering for next time (what they were shopping for, a preferred shop,
their name) that hasn't been saved yet, ask ONCE before they go: "Should I remember this
for next time you call?" If they say yes, save it (following the CONSENT RULE below) —
then say goodbye. If they say no or don't respond clearly, just say goodbye normally.
Don't ask this every single turn — once, near the end of the call, is enough.

CONSENT RULE (hard rule, never skip it):
Before calling save_caller_info, you must first tell the caller in their own words that you
would like to remember this for next time, and get a clear "yes". If they say no, or don't
clearly agree, do NOT call save_caller_info for that information. You may still continue
helping them normally.

GREETING RETURNING CALLERS
If you were given an existing record for this caller at the start of the conversation,
greet them by name and refer naturally to what you last discussed, e.g. "Namaste Ramesh,
last time you ordered 2kg of atta from Sharma Kirana. Would you like the same again?"

GREETING NEW CALLERS
If no record was found for this caller, introduce yourself as Bazaar Mitra, briefly explain
what you can help with, and ask for their name so you can address them personally during
the call — e.g. "I'm Bazaar Mitra. Before we start, may I know your name?" Do this as part
of your very first greeting, not later. Simply asking for and using their name during the
call does NOT need consent — the CONSENT RULE only applies when you want to save it to the
database for next time (see PROACTIVELY OFFER TO REMEMBER above).

CATALOGUE & ORDER TOTALS
You have real catalogue data through two tools — this is now your ONLY source of truth for
prices and stock. Never state a price, stock status, or order total from memory or a guess.

- lookup_products: call this as soon as a caller names a product or category they're
  interested in (e.g. "mouse", "atta", "notebooks"), even before they've mentioned a shop
  or quantity, so you can tell them what's actually available and what it costs.
- compute_order_total: call this once the caller has given you specific products AND
  quantities (e.g. "2kg atta and a wireless mouse"). This gives a PRICE ESTIMATE only —
  it never places or confirms an order (you can never confirm an order yourself, per the
  GUARDRAILS below).

Both tools return an "as_of" date. Mention it naturally when you quote a price or total
(e.g. "as of Aug 9, that's ₹449") so the caller knows how fresh the number is — you don't
need to repeat it on every single sentence, once per quote is enough.

If a tool returns ok: False, the catalogue service is unavailable. Say so plainly and do
NOT guess a price or stock number instead — e.g. "I can't reach our price list right now,
please try again in a bit or contact the shop directly." Treat this like the escalation
script below.

If compute_order_total returns items in "unresolved" (not found, or not enough stock),
tell the caller specifically which items and why — don't just total up what did resolve
and stay silent about the rest.

KNOWLEDGE
You can:
- Explain products.
- Recommend products based on user needs.
- Compare features.
- Look up real prices and stock via lookup_products, and compute order estimates via
  compute_order_total (see CATALOGUE & ORDER TOTALS above).
- Help users understand delivery options if provided.
- Help locate nearby businesses if information is available.

You cannot:
- State a price or stock status without calling lookup_products or compute_order_total.
- Invent delivery dates.
- Confirm an order unless the seller has actually confirmed it.
- Pretend to access live databases.

LANGUAGE
Always mirror the user's language.
If the user mixes Hindi and English, respond in the same style.
If they speak only Hindi, reply in Hindi.
If they speak only English, reply in English.
Use simple conversational language suitable for voice conversations.
Also the text of the response should be in the same language as the user.
Always write every language in its own native script.
Hindi → Devanagari (नमस्ते), never romanized (never "namaste").
Same rule for all non-English languages.

OUTBOUND CALLS
Sometimes YOU are calling the customer, not the other way around — you'll be told this
explicitly in your opening instructions for that call, along with why you're calling. This
is different from an inbound call: the person didn't ask for this call and doesn't know who
you are yet, so open carefully.

Within your very first two sentences, before anything else, you MUST say:
1. Who is calling — Bazaar Mitra — and which shop it's on behalf of, if you know it.
2. Why you're calling, specifically (e.g. confirming a particular order).
3. That they can ask you to stop calling at any time and you'll comply immediately.

Do not wait for them to speak first on an outbound call — you called them, so you open.

If at ANY point — opening or later — the caller says something like "stop calling me",
"don't call again", "remove my number", or similar: say a brief goodbye acknowledging it,
then call opt_out_of_calls. Do this immediately, no matter what else is happening in the
call. This applies on inbound calls too if they ask you to stop calling them in future.

GUARDRAILS

Never:
- Confirm an order unless it has actually been placed.
- Claim a product is in stock without verified information.
- Promise a delivery date.
- Make up prices.
- Pretend to be a human employee.
- Collect passwords, OTPs, PINs, UPI PINs, or payment credentials.
- Ask for sensitive financial information.
- Save any information without first asking and getting a clear "yes" (see CONSENT RULE).
- Continue calling, or call again, someone who has asked you to stop (see OUTBOUND CALLS).
- Create a human escalation without asking and getting a clear "yes" first (see
  HUMAN ESCALATION below).

If asked something outside your capabilities, politely say in user's language of last message:

"I'm sorry, I can't verify that information. Please contact the seller directly for confirmation."

Escalation Script (for things that just aren't worth a human ticket)

If the customer asks about delivery confirmation, or anything else you genuinely can't
verify that ISN'T one of the two situations in HUMAN ESCALATION below, say in the
user's language of last message:

"I can't verify that information myself. Please contact the shop or customer support for confirmation."

HUMAN ESCALATION
For exactly two situations, don't just give the line above — actively create a request
for a human to follow up, using the create_escalation tool:
1. A payment, refund, or order dispute — the caller says they were charged wrong, wants
   a refund, or disputes what an order should have cost or contained.
2. The caller explicitly asks to speak to a human, the shop owner, or says something
   like "I don't want to talk to a bot" / "let me talk to a person".

Nothing else should create an escalation — for anything else unresolved, use the
Escalation Script above instead.

ESCALATION CONSENT RULE (hard rule, never skip it):
Before calling create_escalation, tell the caller PLAINLY what you're about to send —
e.g. "I'll pass along your name, what happened, and how urgent it seems, so someone can
call you back — is that okay?" — and get a clear "yes". If they say no, do NOT call
create_escalation; give them the Escalation Script line instead.

When you do create one, gather these (leave genuinely unknown ones as "unknown" —
never invent them):
- who needs help: their name if you have it, and a number to reach them if they want a
  callback
- what happened: a short, factual summary in your own words — not a transcript
- what you already checked: e.g. "looked up the order in the catalogue — atta was
  ₹42/kg as of Aug 9" or "no matching order found in the catalogue"
- urgency: low / medium / high — your judgment from how they describe it
- their language preference
- how they'd like to be followed up with (call back, any time is fine, a specific time)
NEVER include a password, OTP, PIN, or account number in any of this — you should never
have collected those from the caller in the first place.

After create_escalation succeeds, it gives you a reference ID — tell the caller that ID
and what happens next, honestly. E.g. "Someone from the shop will follow up with you —
I can't promise exactly when, but here's your reference: ESC-0004." Never promise an
immediate reply unless you actually know that's true.

STYLE
- Speak naturally.
- Keep responses under 20 words whenever possible.
- Ask one question at a time.
- Be polite and friendly.
- Never overwhelm the user with long explanations.
- If the user is silent, gently ask if they are still there.
"""


class Assistant(Agent):
    def __init__(self, caller_id: str) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)
        # user_id used to look the caller up in / save the caller to the database.
        # Derived once per session from the room/participant (see get_caller_id below).
        self.caller_id = caller_id

        # ---- Day 8: call outcome tracking ----
        # A call is "successful" if the caller found a real product, priced a real
        # order, or got escalated to a human — i.e. their enquiry was actually
        # completed, not necessarily self-served. Set by lookup_products,
        # compute_order_total, and create_escalation below; read by
        # record_call_outcome() at the end of the call. See README-day8.md.
        self.outcome_achieved = False
        self.outcome_reason: Optional[str] = None
        self.started_at = datetime.now(timezone.utc)

    def _mark_outcome(self, reason: str) -> None:
        if not self.outcome_achieved:
            self.outcome_achieved = True
            self.outcome_reason = reason

    @function_tool
    async def lookup_caller_history(self, context: RunContext) -> dict:
        """Look up whether this caller has spoken with Bazaar Mitra before.

        You already receive a summary of what's known about this caller at the start of
        the conversation, so you normally do NOT need to call this. Only call it later in
        the conversation, after the caller has said something, if you need to re-confirm
        or refresh what's on file (e.g. they ask what you remember about them).

        Returns a dict describing what is on file (name, language_preference, facts,
        last_interaction), or {"found": False} if there is no record for this caller yet.
        """
        user = await db.get_user(self.caller_id)
        if user is None:
            logger.info(f"No existing record for caller {self.caller_id}")
            return {"found": False}
        logger.info(f"Found existing record for caller {self.caller_id}")
        return {"found": True, **user}

    @function_tool
    async def save_caller_info(
        self,
        context: RunContext,
        name: Optional[str] = None,
        language_preference: Optional[str] = None,
        facts: Optional[dict[str, Any]] = None,
    ) -> str:
        """Save or update what you know about this caller, for future conversations.

        Only call this AFTER you have told the caller you'd like to remember this and they
        have clearly agreed. Never call this to store passwords, OTPs, PINs, UPI PINs, or
        other payment/account credentials.

        Args:
            name: The caller's name, if they shared it and agreed you can remember it.
            language_preference: Preferred language/locale, e.g. "hi-IN" or "en-IN".
            facts: Local Commerce facts to remember, e.g. {"past_orders": "2kg atta,
                1L mustard oil", "usual_quantities": "2kg atta weekly",
                "preferred_delivery_slot": "evening", "preferred_shop": "Sharma Kirana"}.
                New values are merged with what is already saved, not replaced.
        """
        await db.save_user(
            self.caller_id,
            name=name,
            language_preference=language_preference,
            facts=facts,
        )
        logger.info(f"Saved info for caller {self.caller_id}: name={name}, facts={facts}")
        return "saved"

    @function_tool
    async def identify_caller(self, context: RunContext, phone_number: str) -> dict:
        """Identify (or start tracking) this caller by a phone number they give you.

        Use this when you don't already recognize the caller from the start-of-call
        context and it would help to recognize them on future calls — for example, they
        ask "do you remember me?", or they want you to remember details for next time.
        Ask for their phone number first, get it, then call this tool with it.

        This makes the phone number the caller's stable ID for the rest of THIS call: any
        later save_caller_info call will save under this phone number, and if a record
        already existed under this number (from a previous call), you'll get it back here.

        Args:
            phone_number: The phone number the caller told you, digits as spoken/heard.
        """
        normalized = normalize_phone(phone_number)
        if not normalized:
            return {"found": False, "error": "that didn't look like a valid phone number"}

        self.caller_id = normalized
        user = await db.get_user(normalized)
        if user is None:
            logger.info(f"No existing record when identifying caller by phone {normalized}")
            return {"found": False}
        logger.info(f"Identified returning caller by phone {normalized}")
        return {"found": True, **user}

    @function_tool
    async def lookup_products(
        self, context: RunContext, query: str, shop: Optional[str] = None
    ) -> dict:
        """Look up real price and stock for a product or category in Bazaar Mitra's
        catalogue.

        ALWAYS call this before telling a caller a specific price or whether something is
        in stock — never guess or state a remembered price. Call it as soon as the caller
        names a product or category (e.g. "mouse", "atta", "notebooks"), even before they
        mention a shop or quantity.

        If the tool result has ok: False, the catalogue service is unavailable right now —
        tell the caller plainly and do not invent a price or stock number instead.

        Args:
            query: The product name or category the caller is asking about, e.g.
                "wireless mouse", "atta", "stationery". Partial names are fine.
            shop: The shop name to filter to, only if the caller named one specifically.
        """
        result = await catalogue.lookup_products(query, shop)
        if result.get("ok") and result.get("matches"):
            self._mark_outcome("product_found")
        return result

    @function_tool
    async def compute_order_total(self, context: RunContext, items: list[dict]) -> dict:
        """Compute a price estimate for specific products and quantities, using real
        catalogue prices and stock. Never calculate or state a total yourself without
        calling this.

        Call this once the caller has told you specific products AND quantities they want
        (e.g. "2kg atta and a wireless mouse"). This gives a PRICE ESTIMATE only — it does
        NOT place or confirm an order (you can never confirm an order yourself).

        If the tool result has ok: False, the catalogue service is unavailable right now —
        tell the caller plainly and do not invent numbers instead. If it lists items under
        "unresolved", tell the caller specifically which items couldn't be priced and why
        (not found, or not enough stock) rather than silently leaving them out.

        Args:
            items: A list of items, each like {"product": "atta", "quantity": 2, "shop":
                "Sharma Kirana"}. "shop" is optional per item — omit it to automatically
                use the cheapest shop that has it. "quantity" is in the product's natural
                unit (kg, litre, piece, pack, etc.) as returned by lookup_products.
        """
        result = await catalogue.compute_order_total(items)
        if result.get("ok") and result.get("line_items"):
            self._mark_outcome("order_priced")
        return result

    @function_tool
    async def opt_out_of_calls(self, context: RunContext) -> None:
        """Mark this caller as do-not-call and end the call.

        Call this if the caller says something like "stop calling me", "don't call
        again", or "remove my number" — on an outbound call or an inbound one. Before
        calling this, say a brief goodbye acknowledging you won't call again; the call
        ends right after you finish speaking, so say that goodbye in the same turn.
        Future outbound calls to this caller should be skipped after this.
        """
        await db.save_user(self.caller_id, facts={"do_not_call": True})
        logger.info(f"Caller {self.caller_id} opted out of future outbound calls")
        await context.wait_for_playout()  # let the goodbye finish playing first
        await _hangup_call()

    @function_tool
    async def create_escalation(
        self,
        context: RunContext,
        reason: str,
        what_happened: str,
        what_agent_checked: str,
        urgency: str,
        caller_language: str,
        preferred_follow_up: str,
    ) -> dict:
        """Create a request for a human to follow up with this caller.

        Only call this for (1) a payment, refund, or order dispute, or (2) the caller
        explicitly asking to speak to a human or the shop owner — and only AFTER you've
        told the caller what you're about to send and gotten a clear "yes" (see the
        ESCALATION CONSENT RULE). Never put a password, OTP, PIN, or account number in
        any argument here.

        Args:
            reason: Short label for why: "payment_or_order_dispute" or "requested_human".
            what_happened: A short, factual summary in your own words — not a transcript.
                E.g. "Caller says they were charged for 2kg atta but only received 1kg."
            what_agent_checked: What you already looked into, e.g. "Looked up atta in
                the catalogue — ₹42/kg as of Aug 9 — but can't see delivered quantities."
            urgency: "low", "medium", or "high" — your judgment from how they describe it.
            caller_language: The language the caller has been speaking, e.g. "Hindi".
            preferred_follow_up: How they'd like to be followed up with, e.g. "call back
                on this number", "any time is fine", "evenings only".
        """
        existing = await db.get_user(self.caller_id)
        caller_name = existing.get("name") if existing else None

        reference_id = await escalations.create_escalation(
            caller_id=self.caller_id,
            caller_name=caller_name,
            reason=reason,
            what_happened=what_happened,
            what_agent_checked=what_agent_checked,
            urgency=urgency,
            caller_language=caller_language,
            preferred_follow_up=preferred_follow_up,
        )
        logger.info(f"Created escalation {reference_id} for caller {self.caller_id}: {reason}")
        self._mark_outcome("escalated")
        return {"reference_id": reference_id}

    # To add more tools, use the @function_tool decorator, following the pattern above.


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


def get_caller_id(ctx: JobContext) -> str:
    """
    Derive a stable id for the caller so we can look them up again on their next call.
    For SIP/telephony calls this prefers the caller's phone number (stable across calls
    from the same phone). Falls back to the participant identity, then the room name.

    NOTE: for web/browser test clients (e.g. a playground that assigns a random
    identity like "voice_assistant_user_1234" on every connection), this fallback is
    NOT stable across calls — that's expected, since there's no phone number to key
    off of. See the identify_caller tool below for how the agent recovers a stable
    identity in that situation, by asking the caller directly.
    """
    for participant in ctx.room.remote_participants.values():
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            phone = participant.attributes.get("sip.phoneNumber") or participant.attributes.get(
                "sip.trunkPhoneNumber"
            )
            if phone:
                return phone
        if participant.identity:
            return participant.identity
    # Last resort: no participant found yet, key off the room name.
    return ctx.room.name


def normalize_phone(phone_number: str) -> str:
    """Reduce a phone number to its last 10 digits, so the same number given with or
    without a country code across two calls (e.g. '+91 98765 43210' vs '9876543210')
    still resolves to the same caller record. Tuned for Indian 10-digit mobile numbers."""
    digits = "".join(ch for ch in phone_number if ch.isdigit())
    return digits[-10:] if len(digits) >= 10 else digits


async def _hangup_call() -> None:
    """Ends the call for everyone by deleting the current room."""
    ctx = get_job_context()
    if ctx is None:
        return
    await ctx.delete_room()


async def record_call_outcome(assistant: "Assistant", call_id: str, call_type: str) -> None:
    """
    Day 8: called once per call, when it ends, to record whether it reached the
    track's success condition — see Assistant.__init__ for what counts as success.
    """
    outcome = "success" if assistant.outcome_achieved else "failed"
    reason = assistant.outcome_reason or "no_resolution"
    await call_stats.record_call(
        call_id=call_id,
        caller_id=assistant.caller_id,
        call_type=call_type,
        outcome=outcome,
        reason=reason,
        started_at=assistant.started_at.isoformat(),
    )
    logger.info(f"Recorded call outcome: {call_id} ({call_type}) -> {outcome} ({reason})")


def build_outbound_opening(reason: Optional[str], call_context: dict, caller_record: Optional[dict]) -> str:
    """
    Build the instructions for the very first thing the agent says on an OUTBOUND call.
    Encodes the Day 6 rule: within the first two sentences, say who's calling, why, and
    how to make it stop — before anything else, and without waiting for the caller to
    speak first.
    """
    name = (caller_record or {}).get("name")
    name_clause = f" ({name})" if name else ""
    shop = call_context.get("shop", "your local shop")

    if reason == "order_confirmation":
        order_summary = call_context.get("order_summary", "a recent order")
        return f"""
        This is an OUTBOUND call you initiated — the person did not call you and doesn't
        know who you are yet, so open carefully. Do NOT wait for them to speak first.

        Within your very first two sentences, before anything else, say:
        1. That this is Bazaar Mitra, calling on behalf of {shop}.
        2. Why you're calling: to confirm their order — {order_summary}.
        3. That they can ask you to stop calling at any time and you will (via
           opt_out_of_calls).

        Example shape (adapt naturally to the caller's likely language{name_clause}):
        "Namaste, this is Bazaar Mitra calling on behalf of {shop} about your order —
        {order_summary}. If you'd like me to stop calling, just say so anytime. Do you
        have a moment to confirm this order?"

        After that opening, listen to their response. You still can't yourself declare
        the order placed — relay their answer; {shop} finalizes it on their end.
        """

    # Generic fallback for any other outbound reason value.
    reason_phrase = reason or "a shopping-related update"
    return f"""
    This is an OUTBOUND call you initiated — the person did not call you and doesn't know
    who you are yet, so open carefully. Do NOT wait for them to speak first.

    Within your very first two sentences, before anything else, say who is calling
    (Bazaar Mitra, on behalf of {shop}), why you're calling ({reason_phrase}), and that
    they can ask you to stop calling at any time and you will (via opt_out_of_calls).
    """


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Make sure the memory database exists before we need it.
    await db.init_db()
    await escalations.init_db()
    await call_stats.init_db()

    # Set up a voice AI pipeline using Murf Falcon, Gemini, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3", language="multi"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-3.5-flash-lite",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="Samar", 
                locale="hi-IN",
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Join the room first, so we can see who's calling (inbound) or place our own
    # call (outbound) before building the Assistant and starting the session.
    await ctx.connect()

    room_options = room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=lambda params: (
                noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC()
            ),
        ),
    )

    # ---- Detect an agent-initiated OUTBOUND call ----
    # Outbound calls are triggered by dispatching this agent with a phone number (and
    # why we're calling) in the job metadata — see trigger_outbound_call.py. Inbound
    # (phone or web) dispatches never set this, so dial_info stays empty for them and
    # everything below is skipped in favor of the existing inbound flow.
    dial_info: dict = {}
    if ctx.job.metadata:
        try:
            dial_info = json.loads(ctx.job.metadata)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Could not parse job metadata as JSON: {ctx.job.metadata!r}")

    phone_number = dial_info.get("phone_number")

    if phone_number:
        # ---------------- OUTBOUND CALL ----------------
        call_reason = dial_info.get("reason")
        call_context = dial_info.get("context") or {}

        caller_id = normalize_phone(phone_number)
        ctx.log_context_fields["caller_id"] = caller_id

        if not OUTBOUND_TRUNK_ID:
            logger.error("LIVEKIT_OUTBOUND_TRUNK_ID is not set — cannot place outbound call")
            ctx.shutdown()
            return

        caller_record = await db.get_user(caller_id)
        if (caller_record or {}).get("facts", {}).get("do_not_call"):
            logger.info(f"Skipping outbound call to {caller_id}: marked do-not-call")
            ctx.shutdown()
            return

        sip_participant_identity = phone_number
        try:
            # This call blocks (wait_until_answered=True) until the phone is actually
            # picked up, so we never start the session — and never speak — into a
            # still-ringing or unanswered call.
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=OUTBOUND_TRUNK_ID,
                    sip_call_to=phone_number,
                    participant_identity=sip_participant_identity,
                    wait_until_answered=True,
                )
            )
            logger.info(f"Outbound call to {phone_number} answered")
        except api.SipCallError as e:
            # e.g. USER_REJECTED (486/603), USER_UNAVAILABLE (408/480), SIP_TRUNK_FAILURE (5xx)
            logger.warning(f"Outbound call to {phone_number} failed: {e.sip_status_code} {e.sip_status}")
            ctx.shutdown()
            return

        participant = await ctx.wait_for_participant(identity=sip_participant_identity)

        outbound_assistant = Assistant(caller_id=caller_id)
        ctx.add_shutdown_callback(
            lambda: record_call_outcome(outbound_assistant, call_id=ctx.room.name, call_type="outbound")
        )

        await session.start(
            agent=outbound_assistant,
            room=ctx.room,
            participant=participant,
            room_options=room_options,
        )

        # Outbound calls speak first — the callee didn't ask for this call, so we open
        # with who/why/how-to-stop rather than waiting for them to say something.
        opening_instructions = build_outbound_opening(call_reason, call_context, caller_record)
        await session.generate_reply(instructions=opening_instructions)
        return

    # ---------------- INBOUND CALL (phone or web) ----------------
    caller_id = get_caller_id(ctx)
    ctx.log_context_fields["caller_id"] = caller_id

    # Look the caller up ourselves, in plain Python, BEFORE the first generate_reply.
    #
    # Why not just let the LLM call the lookup_caller_history tool as its first action?
    # generate_reply(instructions=...) does not add anything to the chat history, so at
    # the very start of a session there is no "user" turn in history yet. If the model's
    # first output is a tool call at that point, Gemini rejects the follow-up request with
    # a 400 ("function call turn comes immediately after a user turn or after a function
    # response turn") because a function-call turn appeared with nothing before it.
    # Doing the lookup here sidesteps that entirely and is also more reliable, since the
    # greeting no longer depends on the model remembering to call the tool.
    caller_record = await db.get_user(caller_id)

    # Start the session, which initializes the voice pipeline and warms up the models
    inbound_assistant = Assistant(caller_id=caller_id)
    ctx.add_shutdown_callback(
        lambda: record_call_outcome(inbound_assistant, call_id=ctx.room.name, call_type="inbound")
    )
    await session.start(
        agent=inbound_assistant,
        room=ctx.room,
        room_options=room_options,
    )

    known_name = caller_record.get("name") if caller_record else None

    if caller_record is None:
        greeting_instructions = """
        This caller has no record on file — you have never spoken with them before.
        Introduce yourself as Bazaar Mitra, explain briefly what you can help with, and
        ask for their name so you can address them personally during this call (this is
        just for the conversation — do not ask about saving it yet). Then ask how you
        can assist today.
        """
    elif not known_name:
        # We recognize this caller (facts on file) but never got a name from them.
        greeting_instructions = f"""
        This caller has spoken with you before, but you don't have a name on file for
        them. Here is what is on file — use it naturally, do not read it out as a list:
        - Language preference: {caller_record.get("language_preference") or "unknown"}
        - Facts: {caller_record.get("facts") or {}}
        - Last interaction: {caller_record.get("last_interaction") or "unknown"}

        Welcome them back, briefly refer to what you last discussed (e.g. their last
        order or preferred shop), ask for their name so you can address them personally,
        then ask how you can help today.
        """
    else:
        greeting_instructions = f"""
        This caller has spoken with you before. Here is what is on file for them —
        use it naturally, do not read it out as a list:
        - Name: {known_name}
        - Language preference: {caller_record.get("language_preference") or "unknown"}
        - Facts: {caller_record.get("facts") or {}}
        - Last interaction: {caller_record.get("last_interaction") or "unknown"}

        Welcome them back by name and briefly refer to what you last discussed (e.g.
        their last order or preferred shop), then ask how you can help today.
        """

    await session.generate_reply(instructions=greeting_instructions)

if __name__ == "__main__":
    cli.run_app(server)