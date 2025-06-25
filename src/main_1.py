import asyncio
import logging
import functools
from typing import Dict, Any
import os
import json

from dotenv import load_dotenv
from openai import AsyncClient
from livekit import agents
from livekit.agents import (
    cli,
    AgentSession, 
    RoomInputOptions, 
    RoomOutputOptions,
    WorkerOptions,
)
from livekit.agents.types import NOT_GIVEN
from livekit.plugins import aws, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel


from core.multi_agent import UserData, TriageAgent, SalesAgent, ServiceAgent
from core.agent import VirtualAgent
from core.callbacks import metrics_callback, shutdown_callback
from core.vision import video_processing_loop
from common.tools import (
    get_product_details,
    auth_user
)
from utils.config import ConfigManager
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


def prewarm(proc: agents.JobProcess, config: Dict[str, Any]):
    """Prewarms resources needed by the agent, VAD etc."""
    logger.info("Prewarming VAD...")    
    vad_config = config['vad']

    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=vad_config['min_speech_duration'],
        min_silence_duration=vad_config['min_silence_duration'],
        prefix_padding_duration=vad_config['prefix_padding_duration'],
        max_buffered_speech=vad_config['max_buffered_speech'],
        activation_threshold=vad_config['activation_threshold'],
        force_cpu=vad_config['force_cpu'],
        sample_rate=vad_config['sample_rate']
    )
    logger.info("VAD prewarmed successfully.")


async def entrypoint(ctx: agents.JobContext, config: Dict[str, Any]):
    """
    The main entrypoint for the agent job
    """
    # Setup initial logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "job_id": ctx.job.id
    } 
    logger.info(f"Agent entrypoint started. Context: {ctx.log_context_fields}")

    await ctx.connect()
    logger.info("Successfully connected to room.")

    # Create shared state dictionary for inter-task communication
    shared_state: Dict[str, Any] = {
        "options": {},
        "selected_option": None,
        "display_options": False,
        "select_option": False
    }

    #  Retrieve the pre-warmed VAD from context's userdata
    vad = ctx.proc.userdata.get("vad")
    if not vad:
        logger.error("VAD not found in process userdata. Exiting.")
        return

    # Initialize UserData with the shared VAD instance
    userdata = UserData(vad=vad, ctx=ctx)

    # Initialize all agent instances
    triage_agent = TriageAgent(config, shared_state, vad)
    sales_agent = SalesAgent(config, shared_state, vad)
    service_agent = ServiceAgent(config, shared_state, vad)

    # Register all agents in the UserData.personas dictionary
    userdata.personas.update({
        "triage": triage_agent,
        "sales": sales_agent,
        "service": service_agent
    })

    # Create the AgentSession with the UserData
    session = AgentSession[UserData](userdata=userdata)
    logger.info("AgentSession created.")

    # Start the video processing loop if configured
    video_task: asyncio.Task | None = None
    vision_config = config['vision']

    if vision_config['use']:
        logger.info("Starting video processing loop...")
        video_task = asyncio.create_task(video_processing_loop(ctx, shared_state, vision_config['video_frame_interval']))

    # Setup metrics collection
    metrics_callback(session)

    # Wait for a participant to join before starting the session
    logger.info("Waiting for participant to join...")
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant '{participant.identity if participant else 'unknown'}' joined.")

    # Register the shutdown callback 
    ctx.add_shutdown_callback(lambda: shutdown_callback(agent=None, video_task=video_task))
    logger.info("Shutdown callback registered.")

    # Start the agent session
    logger.info("Starting agent session with triage...")
    await session.start(
        agent=triage_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC() if config['agent']['use_background_noise_removal'] else NOT_GIVEN,
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )



def main():
    """Main function that initializes and runs the applicaton."""
    # Configure basic logging BEFORE loading config
    logging.basicConfig(level="INFO", 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])
    initial_logger = logging.getLogger(__name__)
    initial_logger.info("Basic logging configured. Loading configuration...")

    # Load configuration 
    app_config = ConfigManager().load_config()
    initial_logger.info("Configuration loaded.")

    # Load env variables
    load_dotenv(app_config['agent']['env_file'])

    # Setup centralized logging using the dedicated function
    setup_logging(config=app_config, project_root=ConfigManager().project_root)

    # Now, get the properly configured logger for the main module
    logger = logging.getLogger(__name__) # Re-get logger after setup
    logger.info("Centralized logging configured. Starting LiveKit Agent application...")

    # Create a partial function that includes the config
    entrypoint_with_config = functools.partial(entrypoint, config=app_config)
    prewarm_with_config = functools.partial(prewarm, config=app_config)

    # Define worker options using loaded config
    worker_config = app_config['worker']
    worker_options = WorkerOptions(
        entrypoint_fnc=entrypoint_with_config, 
        prewarm_fnc=prewarm_with_config,
        job_memory_warn_mb=worker_config['job_memory_warn_mb'],
        load_threshold=worker_config['load_threshold'],
        job_memory_limit_mb=worker_config['job_memory_limit_mb'],
    )

    # Run the CLI application
    cli.run_app(worker_options)


if __name__ == "__main__":
    main()