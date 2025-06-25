import logging
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    vad,
    stt,
    llm,
    tts,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import RunContext
from livekit.plugins import aws

logger = logging.getLogger(__name__)

# Dummy database for demonstration purposes
class CustomerDatabase:
    def get_or_create_customer(self, first_name, last_name):
        return "cust_12345"

    def add_order(self, customer_id, order):
        return "order_67890"

    def get_customer_order_history(self, first_name, last_name):
        # In a real scenario, this would return a list of orders
        return "Order history: 1. Order #123: Item A, 2. Order #456: Item B"

db = CustomerDatabase()

@dataclass
class UserData:
    """Class to store user data, shared plugins, and agents during a call."""
    vad: vad.VAD
    personas: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None
    ctx: Optional[JobContext] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    customer_id: Optional[str] = None
    current_order: Optional[dict] = None

    def is_identified(self) -> bool:
        return self.first_name is not None and self.last_name is not None

    def reset(self) -> None:
        self.first_name = None
        self.last_name = None
        self.customer_id = None
        self.current_order = None

    def summarize(self) -> str:
        if self.is_identified():
            return f"Customer: {self.first_name} {self.last_name} (ID: {self.customer_id})"
        return "Customer not yet identified."

RunContext_T = RunContext[UserData]

class BaseAgent(Agent):
    session: AgentSession[UserData]

    def __init__(
        self,
        instructions: str,
        stt: stt.STT,
        llm: llm.LLM,
        tts: tts.TTS,
        vad: vad.VAD,
    ) -> None:
        super().__init__(
            instructions=instructions,
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
        )

    async def on_enter(self) -> None:
        """Base logic for when an agent becomes active."""
        agent_name = self.__class__.__name__
        logger.info(f"Entering {agent_name}")

        userdata = self.session.userdata
        if userdata.ctx and userdata.ctx.room:
            await userdata.ctx.room.local_participant.set_attributes({"agent": agent_name})

        # Copy chat history from the previous agent
        if userdata.prev_agent:
            # FIX: Introduce a small delay to allow old resources to be released
            await asyncio.sleep(0.2)

            # Add a system message to summarize the transfer for the new agent
            system_message = f"You have just been transferred to the conversation. The previous agent was {userdata.prev_agent.__class__.__name__}. Briefly introduce yourself and continue the conversation based on the history."
            
            chat_ctx = self.chat_ctx.copy()
            
            # Copy items from previous agent
            items_copy = userdata.prev_agent.chat_ctx.items
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in items_copy if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)
            
            # Add the new system message
            chat_ctx.add_message(role="system", content=system_message)
            
            await self.update_chat_ctx(chat_ctx)

    def _truncate_chat_ctx(
        self,
        items: list,
        keep_last_n_messages: int = 6,
        keep_system_message: bool = False,
        keep_function_call: bool = False,
    ) -> list:
        def _valid_item(item) -> bool:
            if not keep_system_message and isinstance(item, llm.ChatMessage) and item.role == "system":
                return False
            if not keep_function_call and isinstance(item, (llm.FunctionCall, llm.FunctionCallOutput)):
                return False
            return True

        new_items = []
        for item in reversed(items):
            if _valid_item(item):
                new_items.append(item)
            if len(new_items) >= keep_last_n_messages:
                break
        new_items = new_items[::-1]

        while new_items and isinstance(new_items[0], (llm.FunctionCall, llm.FunctionCallOutput)):
            new_items.pop(0)

        return new_items

    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> Agent:
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.personas[name]
        userdata.prev_agent = current_agent
        return next_agent

class TriageAgent(BaseAgent):
    def __init__(
        self,
        config: Dict[str, Any],
        shared_state: Dict[str, Any],
        vad_instance: vad.VAD
    ) -> None:
        agent_config = config['agent']['triage']
        transcribe_config = config['stt']['aws_transcribe']
        llm_config = config['llm']['aws_bedrock']
        polly_config = config['tts']['aws_polly']
        
        super().__init__(
            instructions=agent_config['instructions'],
            stt=aws.STT(language=transcribe_config['language']),
            llm=aws.LLM(model=llm_config['model'], temperature=llm_config['temperature']),
            tts=aws.TTS(voice=agent_config['voice'], speech_engine=polly_config['speech_engine'], language=polly_config['language']),
            vad=vad_instance
        )
        self.shared_state = shared_state
        self.greeting_message = agent_config['greeting']

    async def on_enter(self):
        await super().on_enter()
        self.session.say(self.greeting_message)


    @function_tool
    async def authenticate_user(self, first_name: str, last_name: Optional[str] = None):
        """Authenticates a user based on their first name and optional last name."""
        userdata: UserData = self.session.userdata
        userdata.first_name = first_name
        userdata.last_name = last_name or ""
        userdata.customer_id = db.get_or_create_customer(first_name, userdata.last_name)
        return f"User {first_name} authenticated successfully."

    @function_tool
    async def transfer_to_sales(self, context: RunContext_T) -> Agent:
        """Transfer the customer to the sales department."""
        await self.session.say("Of course. Let me transfer you to our sales team.")
        return await self._transfer_to_agent("sales", context)

    @function_tool
    async def transfer_to_service(self, context: RunContext_T) -> Agent:
        """Transfer the customer to the customer service department."""
        await self.session.say("I can help with that. Transferring you to customer service now.")
        return await self._transfer_to_agent("service", context)

class SalesAgent(BaseAgent):
    def __init__(
        self,
        config: Dict[str, Any],
        shared_state: Dict[str, Any],
        vad_instance: vad.VAD
    ) -> None:
        # FIX: Use the 'sales' agent configuration
        agent_config = config['agent']['sales']
        transcribe_config = config['stt']['aws_transcribe']
        llm_config = config['llm']['aws_bedrock']
        polly_config = config['tts']['aws_polly']
        super().__init__(
            instructions=agent_config['instructions'],
            stt=aws.STT(language=transcribe_config['language']),
            llm=aws.LLM(model=llm_config['model'], temperature=llm_config['temperature']),
            tts=aws.TTS(voice=agent_config['voice'], speech_engine=polly_config['speech_engine'], language=polly_config['language']),
            vad=vad_instance
        )


    async def on_enter(self):
        await super().on_enter()
        self.session.generate_reply()

    # ... (Order management tools like start_order, add_item_to_order, complete_order)

    @function_tool
    async def transfer_to_triage(self, context: RunContext_T) -> Agent:
        """Transfer the customer back to the main menu (triage)."""
        await self.session.say("No problem, I'll transfer you back to the main menu.")
        return await self._transfer_to_agent("triage", context)

    # FIX: Removed duplicate transfer_to_triage function

    @function_tool
    async def transfer_to_service(self, context: RunContext_T) -> Agent:
        """Transfer the customer to the customer service department."""
        await self.session.say("Let me transfer you to customer service for assistance with that.")
        return await self._transfer_to_agent("service", context)

class ServiceAgent(BaseAgent):
    def __init__(
        self,
        config: Dict[str, Any],
        shared_state: Dict[str, Any],
        vad_instance: vad.VAD
    ) -> None:
        # FIX: Use the 'service' agent configuration
        agent_config = config['agent']['service']
        transcribe_config = config['stt']['aws_transcribe']
        llm_config = config['llm']['aws_bedrock']
        polly_config = config['tts']['aws_polly']
        super().__init__(
            instructions=agent_config['instructions'],
            stt=aws.STT(language=transcribe_config['language']),
            llm=aws.LLM(model=llm_config['model'], temperature=llm_config['temperature']),
            tts=aws.TTS(voice=agent_config['voice'], speech_engine=polly_config['speech_engine'], language=polly_config['language']),
            vad=vad_instance
        )

    async def on_enter(self):
        await super().on_enter()
        self.session.generate_reply()

    # FIX: Replaced order tools with service-appropriate tools
    @function_tool
    async def get_order_history(self):
        """Get the order history for the current customer."""
        userdata: UserData = self.session.userdata
        if not userdata.is_identified():
            return "Please authenticate the customer first using the authenticate_user function."
        return db.get_customer_order_history(userdata.first_name, userdata.last_name)

    @function_tool
    async def process_return(self, order_id: str, item_name: str, reason: str):
        """Process a return for an item from a specific order."""
        if not self.session.userdata.is_identified():
            return "Please authenticate the customer first."
        # In a real system, this would interact with the database
        return f"Return processed for {item_name} from Order #{order_id}. Reason: {reason}."

    @function_tool
    async def transfer_to_sales(self, context: RunContext_T) -> Agent:
        """Transfer the customer to the sales department."""
        await self.session.say("I'll transfer you to our sales team who can help with a new purchase.")
        return await self._transfer_to_agent("sales", context)

    @function_tool
    async def transfer_to_triage(self, context: RunContext_T) -> Agent:
        """Transfer the customer back to the main menu (triage)."""
        await self.session.say("I'll send you back to the main menu.")
        return await self._transfer_to_agent("triage", context)