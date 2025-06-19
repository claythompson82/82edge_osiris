# scripts/mic_test.py

import asyncio
from scripts.mic import mic_chunks  # ✅ Fix import path

async def mic_demo():
    async for chunk in mic_chunks():
        pass  # Just printing the chunk counter from mic.py

if __name__ == "__main__":
    asyncio.run(mic_demo())
