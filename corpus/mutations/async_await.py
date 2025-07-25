"""Async/await syntax."""

import asyncio


async def main():
    await asyncio.sleep(0)
    return 42


asyncio.run(main())
