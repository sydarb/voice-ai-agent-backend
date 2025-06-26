import re
import json
import uuid
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, AsyncIterable

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    ChatMessage,
    vad,
    stt,
    llm,
    tts,
)
from livekit.agents.llm import function_tool, ChoiceDelta
from livekit.agents.voice import RunContext, ModelSettings
from livekit.agents.llm.chat_context import ImageContent 
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
            return f"Customer: {self.first_name} {self.last_name} (ID: {self.customer_id})."
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
        config: Dict[str, Any],
        shared_state: Dict[str, Any]
    ) -> None:
        super().__init__(
            instructions=instructions,
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
        )
        self.config = config
        self.shared_state = shared_state

    async def on_enter(self) -> None:
        """Base logic for when an agent becomes active."""
        agent_name = self.__class__.__name__
        logger.info(f"Entering {agent_name}")

        userdata = self.session.userdata
        if userdata.ctx and userdata.ctx.room:
            await userdata.ctx.room.local_participant.set_attributes({"agent": agent_name})

        system_message = f"You are the {agent_name}. {userdata.summarize()}"
        chat_ctx = self.chat_ctx.copy()

        # Access the .items attribute to check and remove the last message if it's from the assistant.
        # if chat_ctx.items and chat_ctx.items[-1].type =='message' and chat_ctx.items[-1].role == "assistant":
        #     chat_ctx.items.pop()

        # Copy chat history from the previous agent
        if userdata.prev_agent:
            # FIX: Introduce a small delay to allow old resources to be released
            await asyncio.sleep(0.2)

            system_message += f" You have just been transferred to the conversation. The previous agent was {userdata.prev_agent.__class__.__name__}. Briefly introduce yourself and continue the conversation based on the history."
            
            # Copy items from previous agent
            items_copy = self._truncate_chat_ctx(
                userdata.prev_agent.chat_ctx.items, keep_function_call=True
            )
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in items_copy if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)
            
        chat_ctx.add_message(role="system", content=system_message)
        await self.update_chat_ctx(chat_ctx)

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool],
        model_settings: ModelSettings
    ) -> AsyncIterable[llm.ChatChunk]:
        """
        Processes context via LLM. If the response contains 'choose',
        it buffers the response and constructs a composite JSON message.
        Otherwise, it yields the original response.
        """
        # Check the last 5 messages in the context and clean it if it's a composite message
        if chat_ctx.items:
            logger.debug("Checking last 5 contexts for composite message:")
            for i, msg in enumerate(chat_ctx.items[-5:]):
                logger.debug(f"[{i+1}] {msg}")
            
                if isinstance(msg, ChatMessage) and msg.role == 'assistant' and isinstance(msg.content[0], str):
                    try:
                        data = json.loads(msg.content[0])
                        if isinstance(data, dict) and 'spokenResponse' in data and ('ui' in data or 'ui_actions' in data):
                            logger.debug("Composite message found. Removing the ui content from context.")
                            msg.content[0] = data['spokenResponse']
                    except (json.JSONDecodeError, TypeError):
                        # Not a JSON or not the format we expect, ignore.
                        pass

        logger.debug(f"LLM node received context with {len(chat_ctx.items)} items.")

        if self.config['vision']['use']:
            self._process_image(chat_ctx)

        llm_stream = super().llm_node(chat_ctx, tools, model_settings)
        buffered_chunks = [chunk async for chunk in llm_stream]

        # async for chunk in llm_stream:
        #     if chunk.delta and chunk.delta.content:
        #         chunk.delta.content = chunk.delta.content.rstrip()
        #     yield chunk

        if self.shared_state.get('select_option', False):
            logger.info("select_option is True. Constructing message to select a card.")
            
            full_response = "".join(
                [chunk.delta.content for chunk in buffered_chunks if chunk.delta and chunk.delta.content]
            )
            logger.debug(f"LLM Response: {full_response}")
            selection_index = self.shared_state['selected_option'] or 0
            composite_message = {
                "spokenResponse": full_response.lstrip("\n"),
                "ui_actions": {
                    "type": "ui_action",
                    "action": "select_item",
                    "payload": {
                        "index": selection_index
                    }
                }
            }
            choice_delta = ChoiceDelta(content=json.dumps(composite_message))
            
            self.shared_state['selected_option'] = None
            self.shared_state['select_option'] = False
            
            yield llm.ChatChunk(id=str(uuid.uuid4()), delta=choice_delta)

        elif self.shared_state.get('display_options', False):
            logger.info("display_options is True. Constructing composite message with carousel.")
            
            full_response = "".join(
                [chunk.delta.content for chunk in buffered_chunks if chunk.delta and chunk.delta.content]
            )
            logger.debug(f"LLM Response: {full_response}")

            options = self.shared_state.get('options', {})
            composite_message = {
                "spokenResponse": full_response.lstrip("\n"),
                "ui": options
            }
            choice_delta = ChoiceDelta(content=json.dumps(composite_message))
            
            self.shared_state['display_options'] = False
            self.shared_state['options'] = {}

            yield llm.ChatChunk(id=str(uuid.uuid4()), delta=choice_delta)
        
        else:
            for chunk in buffered_chunks:
                yield chunk

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
    

    def _process_image(self, chat_ctx: llm.ChatContext):
        """Checks for vision keywords and adds latest image from shared_state if applicable."""
        # Check if latest_image exists in shared_state
        if 'latest_image' not in self.shared_state:
            logger.warning("No 'latest_image' key found in shared_state")
            return
        latest_image = self.shared_state['latest_image']
        if not latest_image or not chat_ctx.items:
            return
        last_message = chat_ctx.items[-1]

        # Ensure the last message is a user message before processing
        if not isinstance(last_message, ChatMessage) or last_message.role != "user":
            logger.debug(f"Skipping image processing, last message is of type {type(last_message).__name__}")
            return

        if not last_message.content or not isinstance(last_message.content[0], str):
            return

        user_text = last_message.content[0]
        vision_keywords = ['see', 'look', 'picture', 'image', 'visual', 'color', 'this', 'object', 'view', 'frame', 'screen', 'desk', 'holding']

        if any(keyword in user_text.lower() for keyword in vision_keywords):
            logger.info(f"Vision keyword found in '{user_text[:50]}...'. Adding image to context.")
            if not isinstance(last_message.content, list):
                 last_message.content = [last_message.content] 
            last_message.content.append(ImageContent(image=latest_image))
            logger.debug("Successfully added ImageContent to the last message.")

    @staticmethod
    def clean_text(text_chunk: str) -> str:
        """Cleans text by removing special tags, code blocks, markdown, and emojis."""
        cleaned = text_chunk.replace("<think>", "").replace("</think>", "")
        cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"'''(.*?)'''", r'\1', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'(\*\*|__)(.*?)\1', r'\2', cleaned)
        cleaned = re.sub(r'(\*|_)(.*?)\1', r'\2', cleaned)
        cleaned = re.sub(r'`([^`]*)`', r'\1', cleaned)
        cleaned = re.sub(r'\\+\(', '', cleaned)
        cleaned = re.sub(r'\\+\)', '', cleaned)
        cleaned = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+', '', cleaned)
        return cleaned

    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """
        Cleans text stream and delegates to the default TTS node.
        If the text is a composite message, it only speaks the 'spokenResponse' part.
        """
        logger.debug("TTS node received text stream.")
        full_text = "".join([chunk async for chunk in text])
        text_to_speak = full_text

        try:
            parsed_data = json.loads(full_text)
            if 'spokenResponse' in parsed_data and ('ui' in parsed_data or 'ui_actions' in parsed_data):
                text_to_speak = parsed_data['spokenResponse']
                logger.info("Composite message received. Speaking only the 'spokenResponse' part.")
        except (json.JSONDecodeError, TypeError):
            pass

        cleaned_text = self.clean_text(text_to_speak)
        if cleaned_text:
            async def text_stream():
                yield cleaned_text
            async for frame in super().tts_node(text_stream(), model_settings):
                yield frame
            logger.debug("TTS node finished streaming audio frames.")
        else:
            logger.info("No text content left after cleaning for TTS.")


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
            vad=vad_instance,
            config=config,
            shared_state=shared_state
        )
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
        # await self.session.say("Transferring you to our sales team.")
        return await self._transfer_to_agent("sales", context)

    @function_tool
    async def transfer_to_service(self, context: RunContext_T) -> Agent:
        """Transfer the customer to the customer service department."""
        # await self.session.say("Transferring you to customer service now.")
        return await self._transfer_to_agent("service", context)

class SalesAgent(BaseAgent):
    def __init__(
        self,
        config: Dict[str, Any],
        shared_state: Dict[str, Any],
        vad_instance: vad.VAD
    ) -> None:
        agent_config = config['agent']['sales']
        transcribe_config = config['stt']['aws_transcribe']
        llm_config = config['llm']['aws_bedrock']
        polly_config = config['tts']['aws_polly']
        super().__init__(
            instructions=agent_config['instructions'],
            stt=aws.STT(language=transcribe_config['language']),
            llm=aws.LLM(model=llm_config['model'], temperature=llm_config['temperature']),
            tts=aws.TTS(voice=agent_config['voice'], speech_engine=polly_config['speech_engine'], language=polly_config['language']),
            vad=vad_instance,
            config=config,
            shared_state=shared_state
        )


    async def on_enter(self):
        await super().on_enter()
        self.session.say("Hi! I'm Danielle from the Sales Team. How may I be of assistance to you today?")

    # ... (Order management tools like start_order, add_item_to_order, complete_order)

    @function_tool
    async def transfer_to_triage(self, context: RunContext_T) -> Agent:
        """Transfer the customer back to the main menu (triage)."""
        # await self.session.say("Transferring you back to the main desk.")
        return await self._transfer_to_agent("triage", context)

    @function_tool
    async def transfer_to_service(self, context: RunContext_T) -> Agent:
        """Transfer the customer to the customer service department."""
        # await self.session.say("Transferring you to customer service now.")
        return await self._transfer_to_agent("service", context)


URLS = {
    "chair": "https://cdn.decornation.in/wp-content/uploads/2020/07/modern-dining-table-chairs.jpg",
    "table": "https://mywakeup.in/cdn/shop/files/iso_ae625879-0315-46cd-b978-9cb500fd1a32.png?v=1748581246&width=1214",
    "service": "https://img.freepik.com/free-photo/delivery-concept-handsome-african-american-delivery-man-crossed-arms-isolated-grey-studio-background-copy-space_1258-1277.jpg?semt=ais_hybrid&w=740",

}


class ServiceAgent(BaseAgent):
    def __init__(
        self,
        config: Dict[str, Any],
        shared_state: Dict[str, Any],
        vad_instance: vad.VAD
    ) -> None:
        agent_config = config['agent']['service']
        transcribe_config = config['stt']['aws_transcribe']
        llm_config = config['llm']['aws_bedrock']
        polly_config = config['tts']['aws_polly']
        super().__init__(
            instructions=agent_config['instructions'],
            stt=aws.STT(language=transcribe_config['language']),
            llm=aws.LLM(model=llm_config['model'], temperature=llm_config['temperature']),
            tts=aws.TTS(voice=agent_config['voice'], speech_engine=polly_config['speech_engine'], language=polly_config['language']),
            vad=vad_instance,
            config=config,
            shared_state=shared_state
        )
        db_path = f"./summy_db/product_db.json"
        with open(db_path, "r") as f:
            self.product_db = json.load(f)


    async def on_enter(self):
        await super().on_enter()
        self.session.generate_reply(user_input="hi", instructions="greet user by introducing yourself and asking how you can assist. use the context from the previous messages.")

    # --- NEW SERVICE TOOLS ---

    @function_tool
    async def get_user_products(self) -> str:
        """
        Retrieves the names and descriptions for all products currently rented by the user.
        Call this when you need to know what items the user has, especially when they need to identify a product for a service request.
        """
        # In a real system, you would get the specific user's products.
        # Here, we return all products from our dummy database.
        products = [f"{name}: {details['description']}" for name, details in self.product_db.items()]
        return json.dumps(products)

    @function_tool
    async def display_user_products(self, products: List[str]) -> str:
        """
        Displays a list of product names to the user so they can choose one.
        Use this tool after you have matched a visual description to a few potential products from the get_user_products tool.
        """
        logger.info(f"!!! get_product_details called with: {products}")
        
        if not products:
            return "I could not identify any specific products. Can you please provide the product name?"

        # Dummy product data
        all_products = {
            "Product 1": {"title": "Product 1", "description": "This is a great product you should buy.", "imageUrl": URLS['table'], "actionUrl": "https://example.com/product1"},
            "Product 2": {"title": "Product 2", "description": "This is another great product.", "imageUrl": URLS['chair'], "actionUrl": "https://example.com/product2"},
            "Service A": {"title": "Service A", "description": "Our best service offering.", "imageUrl": URLS['service'], "actionUrl": "https://example.com/serviceA"},
        }
        
        product_details = list(all_products.values())

        if product_details:
            self.shared_state["options"] = {
                "type": "carousel",
                "items": product_details
            }
            self.shared_state["display_options"] = True
            
        # The LLM will use this return string to formulate its response to the user.
        return f"Successfully fetched details for the requested products and displayed to the user. "\
                "Do not generate nay response, only just ask to 'choose any one from the provided options'"

    @function_tool
    async def select_option(self, index: int) -> str:
        """
        Confirms the user's numbered choice (by its 1-based index) from a list of options presented to the user.
        Call this tool when the user has made a choice.
        """
        logger.info(f"!!! select_item called with: {index}")

        if not isinstance(index, int) or index <= 0:
            return "Please provide a valid choice (starting from 1)."
    
        zero_based_index = index - 1
        self.shared_state["selected_option"] = zero_based_index
        self.shared_state["select_option"] = True
    
        return f"Selected option number {index}."

    @function_tool
    async def get_product_known_issues(self, product_name: str) -> str:
        """
        Fetches a list of known issues and their potential solutions for a specific product.
        Call this after the user has confirmed the product they are having trouble with.
        """
        product = self.product_db.get(product_name)
        if not product:
            return f"I could not find a product named '{product_name}' in the database."
        return json.dumps(product['issues'])

    @function_tool
    async def check_for_warranty(self, product_name: str) -> bool:
        """
        Checks if the specified product is currently covered under the free repair warranty.
        Returns True if it is under warranty, and False if it is not.
        """
        # Dummy logic: for simulation, we'll say all products are under warranty.
        print(f"Checking warranty for {product_name}...")
        return True

    @function_tool
    async def fetch_and_display_repair_slots(self) -> str:
        """
        Fetches and returns available time slots for a repair appointment.
        Call this tool when a user wants to schedule a repair.
        """
        # Dummy slots for demonstration
        return "We have the following slots available: Tomorrow at 10 AM, tomorrow at 2 PM, or Friday at 11 AM."

    @function_tool
    async def confirm_repair_slot(self, slot: str) -> str:
        """
        Confirms and books a repair appointment for the customer in the specified time slot.
        """
        # In a real system, this would write to a calendar or database.
        return f"Excellent. Your repair appointment has been successfully scheduled for {slot}. You will receive a confirmation email shortly."

    # --- TRANSFER TOOLS ---

    @function_tool
    async def transfer_to_sales(self, context: RunContext_T) -> Agent:
        """Transfer the customer to the sales department."""
        await self.session.say("Transferring you to our sales team.")
        return await self._transfer_to_agent("sales", context)

    @function_tool
    async def transfer_to_triage(self, context: RunContext_T) -> Agent:
        """Transfer the customer back to the main menu (triage)."""
        await self.session.say("Transferring you back to the main desk.")
        return await self._transfer_to_agent("triage", context)