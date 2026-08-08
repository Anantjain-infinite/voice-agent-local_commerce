import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    tokenize,
    room_io,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

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
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

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

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
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

    # Join the room and connect to the user
    await ctx.connect()

    await session.generate_reply(
    instructions="""
    Greet the customer.
    Introduce yourself as Bazaar Mitra.
    Explain briefly what you can help with.
    Ask how you can assist today.
    """
)

if __name__ == "__main__":
    cli.run_app(server)
