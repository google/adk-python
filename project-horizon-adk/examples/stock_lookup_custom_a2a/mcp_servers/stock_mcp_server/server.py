import logging
import asyncio
from mcp.server.fastmcp import FastMCP
import finnhub
import sys
import json
import os
from dotenv import load_dotenv

# Determine the absolute path to the .env file in the project root
# __file__ is the path to the current script (server.py)
# os.path.dirname(__file__) is the directory of server.py
# os.path.join(..., "..", "..") goes up two directories to the project root
project_root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
load_dotenv(project_root_env, override=True) # Load variables from .env file

# --- Configuration ---
# Basic logging setup
logging.basicConfig(level=logging.DEBUG, # <--- Change level to DEBUG for more detail
                    stream=sys.stderr,
                    format='%(asctime)s - %(name)s - %(levelname)s - MCP_SERVER - %(message)s')
logger = logging.getLogger(__name__)

# --- Finnhub Client Initialization ---
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    logger.warning("FINNHUB_API_KEY environment variable not found. Stock price queries will use mock data.")
    
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY or "")
if FINNHUB_API_KEY:
    logger.info(f"Finnhub client initialized with API key: {FINNHUB_API_KEY[:3]}...{FINNHUB_API_KEY[-3:] if len(FINNHUB_API_KEY) > 6 else ''}")
else:
    logger.info("Finnhub client initialized without API key")

# --- FastMCP Server Initialization ---
# Creates a customizable MCP server named "Stock Price Server"
logger.info("Initializing FastMCP server...")
mcp = FastMCP("Stock Price Server") # Name used during initialization

# --- Tool Functions (using decorators) ---

@mcp.tool()
async def get_current_stock_price(symbol: str) -> dict:
    """
    Retrieve the current stock price for the given ticker symbol.
    Returns a dictionary containing the price and currency, or an error message.
    """
    # Log the incoming arguments (part of the MCP request's params)
    logger.debug(f"Tool 'get_current_stock_price' called with args: {{'symbol': '{symbol}'}}")

    # Check MOCK_STOCK_API environment variable
    mock_api_enabled = os.getenv("MOCK_STOCK_API", "False").lower() == "true"

    if mock_api_enabled:
        logger.info("MOCK_STOCK_API is enabled. Returning mock data.")
        result = {
            "symbol": symbol.upper(),
            "price": 123.45,
            "currency": "USD",
            "note": "Mock data (MOCK_STOCK_API enabled)"
        }
        logger.debug(f"Tool 'get_current_stock_price' returning result: {json.dumps(result, indent=2)}")
        return result

    # If MOCK_STOCK_API is not true, proceed with actual API call
    try:
        # Use Finnhub API to get stock quote
        quote = finnhub_client.quote(symbol.upper())
        
        # Check if we have a valid quote
        if quote and 'c' in quote and quote['c'] > 0:
            price = quote['c']  # Current price
            result = {
                "symbol": symbol.upper(),
                "price": round(price, 2),
                "currency": "USD",  # Finnhub uses USD by default
                "previous_close": quote.get('pc', None)  # Previous close price
            }
            logger.info(f"Tool: Successfully found price {result['price']} USD for {symbol} using Finnhub.")
        else:
            # If no valid price in quote, use previous close if available
            if quote and 'pc' in quote and quote['pc'] > 0:
                price = quote['pc']  # Previous close
                result = {
                    "symbol": symbol.upper(),
                    "price": round(price, 2),
                    "currency": "USD",
                    "note": "Using previous close price"
                }
                logger.info(f"Tool: Using previous close price {result['price']} USD for {symbol}.")
            else:
                # No valid data from Finnhub
                logger.warning(f"Tool: No current price data found for symbol {symbol}. Returning mock data.")
                result = {
                    "symbol": symbol.upper(),
                    "price": 123.45,
                    "currency": "USD",
                    "note": "Mock data: Price not found via Finnhub"
                }

    except Exception as e:
        logger.error(f"Tool: Error fetching price for {symbol} from Finnhub: {e}. Returning mock data.", exc_info=False)
        result = {
            "symbol": symbol.upper(),
            "price": 123.45,
            "currency": "USD",
            "note": f"Mock data: Finnhub API error ({type(e).__name__})"
        }

    # Log the result dictionary (which will be the MCP response's result field)
    logger.debug(f"Tool 'get_current_stock_price' returning result: {json.dumps(result, indent=2)}")
    return result


# --- Main Execution ---
if __name__ == "__main__":
    # --- To run the MCP server (original behavior) ---
    logger.info("Starting FastMCP Stock Price Server...")
    mcp.run() # defaults to stdio transport
    logger.info("FastMCP Stock Price Server stopped.")
