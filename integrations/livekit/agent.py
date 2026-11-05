"""A LiveKit voice agent on Crusoe Managed Inference with one function tool.

``python agent.py console`` talks through the local mic; ``python agent.py dev`` joins LiveKit
rooms. Requires the keys in .env.example. Incurs inference and speech charges.
"""

import os

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, RunContext, WorkerOptions, cli, function_tool
from livekit.plugins import cartesia, deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

CRUSOE_BASE_URL = "https://api.inference.crusoecloud.com/v1/"


class Concierge(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly concierge for a GPU cloud. Replies are spoken aloud, so keep "
                "them short and avoid lists or formatting. Use the tool for availability questions."
            )
        )

    @function_tool
    async def gpu_availability(self, context: RunContext, region: str, gpu_type: str) -> str:
        """Check how many GPUs of a type are free in a region right now.

        Args:
            region: Region slug such as "us-east1" or "eu-iceland1".
            gpu_type: GPU model such as "h100" or "b200".
        """
        inventory = {("us-east1", "h100"): 512, ("us-east1", "b200"): 64, ("eu-iceland1", "h100"): 256}
        count = inventory.get((region.lower(), gpu_type.lower()), 0)
        return f"{count} {gpu_type.upper()} GPUs are available in {region}."


async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(
            model=os.environ.get("CRUSOE_MODEL", "zai/GLM-5.3"),
            base_url=CRUSOE_BASE_URL,
            api_key=os.environ["CRUSOE_API_KEY"],
            temperature=0.3,
        ),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )
    await session.start(agent=Concierge(), room=ctx.room)
    await session.generate_reply(instructions="Greet the user and offer to check GPU availability.")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
