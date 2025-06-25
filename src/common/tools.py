import time
import logging
from functools import wraps


logger = logging.getLogger(__name__)


URLS = {
    "chair": "https://cdn.decornation.in/wp-content/uploads/2020/07/modern-dining-table-chairs.jpg",
    "table": "https://mywakeup.in/cdn/shop/files/iso_ae625879-0315-46cd-b978-9cb500fd1a32.png?v=1748581246&width=1214",
    "service": "https://img.freepik.com/free-photo/delivery-concept-handsome-african-american-delivery-man-crossed-arms-isolated-grey-studio-background-copy-space_1258-1277.jpg?semt=ais_hybrid&w=740",

}

def log_tool_call(func):
    """
    A decorator to log the details of an asynchronous tool call, including its
    name, arguments, return value, and execution time.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger.info(f"---- Calling tool: {func.__name__} ----")
        logger.info(f"Arguments: args={args}, kwargs={kwargs}")

        start_time = time.perf_counter()

        result = await func(*args, **kwargs)
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        logger.info(f"Output: {result}")
        logger.info(f"---- Finished tool: {func.__name__} in {execution_time:.4f} seconds ----")

        return result
    return wrapper


@log_tool_call
async def get_product_details(shared_state: dict, product_name: str) -> str:
        logger.info(f"!!! get_product_details called with: {product_name}")
        # Dummy product data
        all_products = {
            "Product 1": {"title": "Product 1", "description": "This is a great product you should buy.", "imageUrl": URLS['table'], "actionUrl": "https://example.com/product1"},
            "Product 2": {"title": "Product 2", "description": "This is another great product.", "imageUrl": URLS['chair'], "actionUrl": "https://example.com/product2"},
            "Service A": {"title": "Service A", "description": "Our best service offering.", "imageUrl": URLS['service'], "actionUrl": "https://example.com/serviceA"},
        }
        
        product_details = list(all_products.values())

        if product_details:
            # Update options data in shared_state
            shared_state["options"] = {
                "type": "carousel",
                "items": product_details
            }
            shared_state["display_options"] = True
            
            return f"Successfully fetched details for the requested products and displayed to the user. "\
                "Do not generate nay response, only just ask to 'choose any one from the provided options'"
        
        else:
            return "Could not find details for the requested products."
        