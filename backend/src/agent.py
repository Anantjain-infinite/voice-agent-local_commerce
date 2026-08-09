import logging
from typing import Any, Optional

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
    tokenize,
    room_io,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import db

logger = logging.getLogger("agent")

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

KNOWLEDGE
You can:
- Explain products.
- Recommend products based on user needs.
- Compare features.
- Explain general shopping information.
- Help users understand delivery options if provided.
- Help locate nearby businesses if information is available.

You cannot:
- Invent prices.
- Invent stock availability.
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

If asked something outside your capabilities, politely say in user's language of last message:

"I'm sorry, I can't verify that information. Please contact the seller directly for confirmation."

Escalation Script

If the customer requests:
- Order confirmation
- Refund approval
- Delivery confirmation
- Live inventory
- Payment issues

say in user's language of last message:

"I can't verify that information myself. Please contact the shop or customer support for confirmation."

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


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Make sure the memory database exists before we need it.
    await db.init_db()

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

    # Join the room and connect to the user first, so we can see who's calling
    # before we build the Assistant and start the session.
    await ctx.connect()

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
    await session.start(
        agent=Assistant(caller_id=caller_id),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
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