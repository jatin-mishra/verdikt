"""Minimal end-to-end example.

    export OPENAI_API_KEY=sk-...
    python examples/quickstart.py
"""
import asyncio

from verdikt import EvalInput, JudgeConfig, Verdikt
import os

from verdikt.core.schemas import ProviderConfig 

# GEMINI_AGENT_MODEL, GEMINI_API_KEY
# ANTHROPIC_AGENT_API_KEY, ANTHROPIC_AGENT_MODEL

async def main() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(
                name="helpfulness",
                type="pointwise",
                model="gemini/" + os.environ.get("GEMINI_AGENT_MODEL"),
                criteria=["Directly answers the question", "No factual errors"],
                threshold=0.7,
            ),
            JudgeConfig(
                name="judge-2",
                type="pointwise",
                model="anthropic/" + os.environ.get("ANTHROPIC_AGENT_MODEL"),
                criteria=["Directly answers the question", "No factual errors"],
                threshold=0.7,
            )
        ],
        providers={
            "gemini" : ProviderConfig(
                api_key=os.environ.get("GEMINI_API_KEY"),
                protocol="gemini"
            ),
            "anthropic" : ProviderConfig(
                api_key=os.environ.get("ANTHROPIC_AGENT_API_KEY"),
                protocol="anthropic"
            )
        }
    )
    print(vd)
    verdict = await vd.evaluate(
        "helpfulness",
        EvalInput(
            input="What is the capital of France?",
            output="The capital of France is Paris.",
        ),
    )
    print("score:    ", verdict.score)
    print("passed:   ", verdict.passed)
    print("reasoning:", verdict.reasoning)
    print("cost usd: ", verdict.meta.cost_usd)


if __name__ == "__main__":
    asyncio.run(main())
