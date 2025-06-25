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
    BackgroundAudioPlayer,
    AudioConfig,
    BuiltinAudioClip,
    function_tool
)
from livekit.agents.types import NOT_GIVEN
from livekit.plugins import aws, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from core.agent import VirtualAgent
from core.callbacks import metrics_callback, shutdown_callback
from core.vision import video_processing_loop
from engine.llm import OpenaiLLM
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
        "display_options": False,
    }

    # Initialize LLM Client using config
    llm_config = config['llm']
    if llm_config['use_local']:
        try:
            llm_client = AsyncClient(api_key=llm_config['ollama']['api_key'], base_url=llm_config['ollama']['base_url'])
            logger.info(f"Initialized LLM Client at {llm_config['ollama']['base_url']}")

        except Exception as e:
            logger.error(f"Failed to initialize LLM Client: {e}")
            raise

        llm_instance = OpenaiLLM(client=llm_client, config=llm_config)
        logger.info(f"Created an OpenAI LLM instance.")

    else:
        llm_instance = aws.LLM(
            model=llm_config['aws_bedrock']['model'],
            temperature=llm_config['aws_bedrock']['temperature']
        )
        logger.info(f"Created an AWS Bedrock LLM instance.")

    # Check if VAD was prewarmed successfully
    vad_instance = ctx.proc.userdata.get("vad")
    if not vad_instance:
        logger.error("VAD not found in process userdata. Exiting.")
        return

    # Setup Speech-to-text and Text-to-speech
    transcribe_config = config['stt']['aws_transcribe']
    stt_instance = aws.STT(
        language=transcribe_config['language'],
        session_id=NOT_GIVEN,
        vocabulary_name=NOT_GIVEN,
        vocab_filter_name=NOT_GIVEN,
        vocab_filter_method=NOT_GIVEN,
    )
    polly_config = config['tts']['aws_polly']
    tts_instance = aws.TTS(
        voice=polly_config['voice'],
        speech_engine=polly_config['speech_engine'],
        language=polly_config['language'],
    )

    # Setup the AgentSession with configured plugins
    session = AgentSession(
        stt = stt_instance,
        llm=llm_instance,
        tts=tts_instance,
        vad=vad_instance,
        turn_detection=MultilingualModel() if config['agent']['use_eou'] else NOT_GIVEN,
    )
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

    # dummy tool
    async def _get_weather_info(location: str) -> str:
        logger.info(f"!!! _get_weather_info called with: {location}")
        if location.lower() == "hyderabad":
            return f"Its always sunny in {location}!"
        elif location.lower() == "london":
            return f"Its currently raining in {location}!"
        else:
            return f"Its bad weather in {location}!"
        
    async def _get_knowledge_bases():
        """
        get list of knowledge bases for available products and services
        """
        db_path = "./database/"
        file_list = '\n'.join([f for f in os.listdir(db_path) if os.path.isfile(os.path.join(db_path, f))]).strip()
        return f"Here is the list of knowledge bases avialable:\n{file_list}"

    async def _get_product_service(knowledge_base: str = "laptop repair store"):
        """
        get list of products and services for a given knowledge base
        """
        db_path = f"./database/{knowledge_base}.json"
        with open(db_path, "r") as f:
            db = json.load(f)
        product_service_list = ["\nProduct/service: {}".format(product_service) for product_service in db]
        return f"Here is the list of products/service offered in knowledge base {knowledge_base}: {''.join(product_service_list).rstrip()}"

    async def _get_issue(product_service: str, knowledge_base: str = "laptop repair store"):
        """
        get list of known problems if present for a given knowledge base and product/service
        """
        db_path = f"./database/{knowledge_base}.json"
        with open(db_path, "r") as f:
            db = json.load(f)
        product_issue_list = ["\nIssue: {}".format(issue['issue']) for issue in db[product_service]]
        return f"Here is the list of issues affecting {product_service} for which data is avialable: {''.join(product_issue_list).rstrip()}"
        
    async def _get_issue_solution(issue: str, product_service: str, knowledge_base: str = "laptop repair store"):
        """
        get the solution for a given product/service's issue from a given knowledge base
        """
        db_path = f"./database/{knowledge_base}.json"
        with open(db_path, "r") as f:
            db = json.load(f)

        for kb_product_issue in db[product_service]:
            if kb_product_issue['issue'].lower().strip() == issue.lower().strip():
                return f"Here is a possible solution for the given issue:\n{kb_product_issue['solution']}"
        return "Couldn't find solution in the stored knowledge base. I will have to use the web to search for a solution"

    async def _get_possible_issue_cause(issue: str, product_service: str, knowledge_base: str = "laptop repair store"):
        """
        get a possible reason for the issue affecting given product/service
        """
        db_path = f"./database/{knowledge_base}.json"
        with open(db_path, "r") as f:
            db = json.load(f)

        for kb_product_issue in db[product_service]:
            if kb_product_issue['issue'].lower().strip() == issue.lower().strip():
                return f"Here is a possible cause for the given issue:\n{kb_product_issue['potential_reason']}"
        return "Couldn't find the issue in the stored knowledge base. I will have to use the web to search for a solution"



    async def _get_product_details(product_name: str) -> str:
        logger.info(f"!!! get_product_details called with: {product_name}")
        # Dummy product data
        all_products = {
            "Product 1": {"title": "Product 1", "description": "This is a great product you should buy.", "imageUrl": "https://via.placeholder.com/250x150?text=Product+1", "actionUrl": "https://example.com/product1"},
            "Product 2": {"title": "Product 2", "description": "This is another great product.", "imageUrl": "https://via.placeholder.com/250x150?text=Product+2", "actionUrl": "https://example.com/product2"},
            "Service A": {"title": "Service A", "description": "Our best service offering.", "imageUrl": "https://via.placeholder.com/250x150?text=Service+A", "actionUrl": "https://example.com/serviceA"},
        }
        
        product_details = list(all_products.values())

        if product_details:
            # Update options data in shared_state
            shared_state["options"] = {
                "type": "carousel",
                "items": product_details
            }
            shared_state["display_options"] = True
            
            return f"Successfully fetched details for the requested products and displayed to the user. Do not generate a response to this, just ask to 'choose any from the below options'"
        
        else:
            return "Could not find details for the requested products."

    # Setup agent instance
    agent = VirtualAgent(
        participant_identity=participant.identity,
        shared_state=shared_state,
        config=config,
        room=ctx.room,
        tools=[
            # function_tool(
            #     _get_weather_info,
            #     name="get_weather_info",
            #     description="Get the weather in a specific location",
            # ),
            # function_tool(
            #     _get_product_details,
            #     name="get_product_details",
            #     description="Get the details for all the products with the provided product name",
            # ),
            function_tool(
                _get_knowledge_bases,
                name="_get_knowledge_bases",
                description="Get list of knowledge bases for available products and services",
            ),
            function_tool(
                _get_product_service,
                name="_get_product_service",
                description="Get the list of products and services available in a given knowledge base",
            ),
            function_tool(
                _get_issue,
                name="_get_issue",
                description="Get the list of known problems if present for a given knowledge base and product/service",
            ),
            function_tool(
                _get_issue_solution,
                name="_get_issue_solution",
                description="Get the solution for a given product/service issue from a given knowledge base",
            ),
            function_tool(
                _get_possible_issue_cause,
                name="_get_possible_issue_cause",
                description="Get a potential reason for the issue affecting the given product/service",
            ),
        ],
    )

    # Register the shutdown callback 
    ctx.add_shutdown_callback(lambda: shutdown_callback(agent, video_task))
    logger.info("Shutdown callback registered.")

    # Start the agent session
    logger.info("Starting agent session...")
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC() if config['agent']['use_background_noise_removal'] else NOT_GIVEN,
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    # Start background audio
    if config['agent']['use_background_audio']:
        background_audio = BackgroundAudioPlayer(
            # play office ambience sound looping in the background
            ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.8),
            # play keyboard typing sound when the agent is thinking
            thinking_sound=[
                AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8),
                AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.7),
            ],
        )
        await background_audio.start(room=ctx.room, agent_session=session)

    # await session.generate_reply(
    #     instructions="Greet the user and offer your assistance."
    # )


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