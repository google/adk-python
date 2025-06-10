"""
Custom tools for the Financial Expert agent, designed to be compatible with
Google ADK (Agent Development Kit) tool requirements. These tools simulate
interactions with financial web searches, SEC EDGAR APIs, and stock price APIs.

ADK Compatibility Notes:
For use with Google ADK, these functions can be:
1. Wrapped with `google.adk.tools.FunctionTool` along with their schema.
2. Used as methods within a custom ADK Tool class.
The key is that they are callable and return JSON-serializable results,
often a dictionary or a string summarizing the outcome. The conceptual schemas
provided in each function's docstring illustrate how they might be defined for ADK.
"""

import json
from typing import Dict, Any, List, Optional

def financial_web_search_tool(
    query: str,
    num_results: int = 3,
    site_filter: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Performs a simulated financial web search.

    # ADK Tool Schema:
    # name: financial_web_search
    # description: Searches financial news and general web for financial topics.
    # parameters:
    #   type: object
    #   properties:
    #     query:
    #       type: string
    #       description: The search query for financial information.
    #     num_results:
    #       type: integer
    #       description: Desired number of search results. Defaults to 3.
    #     site_filter:
    #       type: string
    #       description: Optional specific website to search within (e.g., wsj.com, bloomberg.com).
    #   required:
    #     - query

    Args:
        query: The search query.
        num_results: The number of mock results to generate.
        site_filter: An optional domain to simulate filtering the search.

    Returns:
        A list of dictionaries, where each dictionary represents a mock search result.
    """
    print(f"financial_web_search_tool: Searching for '{query}' "
          f"(num_results={num_results}, site_filter={site_filter or 'N/A'}) (placeholder).")

    results = []
    for i in range(1, num_results + 1):
        site_info = f" from {site_filter}" if site_filter else ""
        results.append({
            "title": f"Placeholder result {i} for '{query}'{site_info}",
            "link": f"http://example.com/financial_news/{query.replace(' ', '_')}_result{i}",
            "snippet": f"This is mock snippet {i} for the financial query '{query}'{site_info}. "
                       "It contains relevant financial keywords and information."
        })
    return results

def sec_edgar_api_tool(
    cik: str,
    filing_type: str = "10-K",
    num_latest_filings: int = 1
) -> List[Dict[str, str]]:
    """
    Simulates fetching company filings from the SEC EDGAR database.

    # ADK Tool Schema:
    # name: fetch_sec_edgar_filings
    # description: Retrieves company filings (e.g., 10-K, 10-Q) from SEC EDGAR.
    # parameters:
    #   type: object
    #   properties:
    #     cik:
    #       type: string
    #       description: The Central Index Key (CIK) of the company.
    #     filing_type:
    #       type: string
    #       description: The type of filing (e.g., "10-K", "10-Q", "8-K"). Defaults to "10-K".
    #     num_latest_filings:
    #       type: integer
    #       description: Number of latest filings to retrieve. Defaults to 1.
    #   required:
    #     - cik

    Args:
        cik: The Central Index Key (CIK) of the company.
        filing_type: The type of filing (e.g., "10-K", "10-Q").
        num_latest_filings: The number of mock latest filings to return.

    Returns:
        A list of dictionaries, where each dictionary represents a mock SEC filing.
    """
    print(f"sec_edgar_api_tool: Fetching {num_latest_filings} latest '{filing_type}' "
          f"for CIK {cik} (placeholder).")

    filings = []
    for i in range(1, num_latest_filings + 1):
        filings.append({
            "cik": cik,
            "filing_type": filing_type,
            "filing_date": f"2023-0{i}-15", # Mock date
            "report_url": f"https://www.sec.gov/Archives/edgar/data/{cik}/filing{i}.html",
            "summary": f"This is a mock summary for filing {i} of type {filing_type} for CIK {cik}."
        })
    return filings

def stock_price_api_tool(
    ticker_symbol: str,
    date: Optional[str] = None # Format YYYY-MM-DD for historical
) -> Dict[str, Any]:
    """
    Simulates fetching the stock price for a given ticker symbol, either current or historical.

    # ADK Tool Schema:
    # name: get_stock_price
    # description: Fetches the current or historical stock price for a ticker symbol.
    # parameters:
    #   type: object
    #   properties:
    #     ticker_symbol:
    #       type: string
    #       description: The stock ticker symbol (e.g., "AAPL", "MSFT").
    #     date:
    #       type: string
    #       format: date
    #       description: Optional date (YYYY-MM-DD) for historical price. If omitted, fetches current price.
    #   required:
    #     - ticker_symbol

    Args:
        ticker_symbol: The stock ticker symbol.
        date: Optional date (YYYY-MM-DD) for historical price. If None, simulates current price.

    Returns:
        A dictionary containing mock stock price information.
    """
    price_type = f"historical price for date {date}" if date else "latest current price"
    print(f"stock_price_api_tool: Fetching {price_type} for ticker '{ticker_symbol}' (placeholder).")

    # Simulate price based on ticker hash for some variability
    base_price = float(hash(ticker_symbol) % 1000) + 50.00
    if date: # Simulate some variation for historical
        base_price *= (1.0 - (hash(date) % 10) / 100.0) # +/- up to 9%

    return {
        "ticker_symbol": ticker_symbol.upper(),
        "price": round(base_price, 2),
        "currency": "USD",
        "date": date if date else "current_realtime", # or a simulated current timestamp
        "type": "historical" if date else "current",
        "volume": int(base_price * 10000) + (hash(ticker_symbol) % 50000),
        "market_cap": base_price * 10**9 # Highly simplified
    }


if __name__ == '__main__':
    print("--- Testing Financial Tools ---")

    print("\n1. Testing financial_web_search_tool:")
    search_results_general = financial_web_search_tool("Q4 earnings forecast for tech sector")
    print(f"General search results: {json.dumps(search_results_general, indent=2)}\n")
    search_results_filtered = financial_web_search_tool(
        query="renewable energy investments",
        num_results=2,
        site_filter="bloomberg.com"
    )
    print(f"Filtered search results: {json.dumps(search_results_filtered, indent=2)}")

    print("\n2. Testing sec_edgar_api_tool:")
    filings_10k = sec_edgar_api_tool(cik="0000320193", filing_type="10-K", num_latest_filings=1) # Apple CIK
    print(f"10-K filings: {json.dumps(filings_10k, indent=2)}\n")
    filings_8k = sec_edgar_api_tool(cik="0000789019", filing_type="8-K", num_latest_filings=2) # Microsoft CIK
    print(f"8-K filings: {json.dumps(filings_8k, indent=2)}")

    print("\n3. Testing stock_price_api_tool:")
    current_price_aapl = stock_price_api_tool(ticker_symbol="AAPL")
    print(f"Current AAPL price: {json.dumps(current_price_aapl, indent=2)}\n")
    historical_price_msft = stock_price_api_tool(ticker_symbol="MSFT", date="2023-06-30")
    print(f"Historical MSFT price: {json.dumps(historical_price_msft, indent=2)}")

    print("\n--- Financial Tools Test Finished ---")
