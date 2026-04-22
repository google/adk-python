import asyncio
from google.adk.tools.function_tool import FunctionTool

async def generate_image(
    prompt: str,
    input_bytes: list[tuple[bytes, str]] | None = None,
) -> dict:
    """Generate an image from a prompt."""
    return {"status": "success"}

async def main():
    try:
        generate_image_tool = FunctionTool(func=generate_image)
        generate_image_tool._get_declaration()
        print("SUCCESS! No validation error.")
    except Exception as e:
        print(f"FAILED! Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
