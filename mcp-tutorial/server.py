from mcp.server import FastMCP

# setup server
mcp = FastMCP("demo")


# setup tools 
@mcp.tool("add")
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a greeting message."""
    return f"Hello, {name}!"


