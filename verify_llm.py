from rich.console import Console

from src.config import settings
from src.pipeline import require_api_key
from src.generator import GeminiGenerator


console = Console()


def main() -> None:
    generator = GeminiGenerator(
        api_key=require_api_key(),
        model=settings.generation_model,
    )
    response = generator.verify_connection()
    console.print(f"Live Gemini call succeeded ({settings.generation_model}): {response}")


if __name__ == "__main__":
    main()
