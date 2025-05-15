from datetime import datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP('Test')


@mcp.tool()
def get_current_time() -> str:
    """Get the current time."""
    return f'current_time: {datetime.now().isoformat()}'


@mcp.tool()
def tavily_search(search_query: str) -> str:
    """Performs a web search using the Tavily API."""
    return f"Results for '{search_query}' from Tavily search."


@mcp.tool()
def get_weather(city: str, metric: str = 'celsius') -> str:
    """Get weather info for a city."""
    return f'It is 23 {metric} in {city}'


@mcp.tool()
def get_current_weather(location: str, unit: str = 'celsius', user: dict = None) -> dict:
    """Get the current weather in a location, customized by user details."""
    if not user or 'name' not in user:
        raise ValueError("User details with 'name' field are required.")

    return {'location': location, 'unit': unit, 'user': user, 'weather': f'Sunny, 22 degrees {unit}'}


@mcp.tool()
def get_user_info(user_id: int, special: str = 'none') -> dict:
    """Retrieve user details by ID."""
    return {'user_id': user_id, 'special': special, 'info': {'name': 'John Doe', 'email': 'john.doe@example.com'}}


@mcp.tool()
def yahoo_finance_search(symbol: str) -> dict:
    """Get current stock information for a given symbol."""
    return {'symbol': symbol, 'price': '123.45', 'currency': 'USD'}


@mcp.tool()
def exa_news_search(query: str, answer: bool = False) -> dict:
    """Search news via Exa API."""
    return {'query': query, 'answer_required': answer, 'results': ['News headline 1', 'News headline 2']}


@mcp.tool()
def my_adder_tool(a: int, b: int) -> str:
    """Takes two integers and returns their sum."""
    return f'sum: {a + b}'


if __name__ == '__main__':
    mcp.run(transport='stdio')
