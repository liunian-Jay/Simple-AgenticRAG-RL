import asyncio
import httpx

async def send_request():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/retrieve",
            json={"query": "What is Python?", "topk": 3, "return_scores": True}
        )
        print(response.status_code, response.text)

async def main():
    tasks = [send_request() for _ in range(500)]  # 50 个并发请求
    await asyncio.gather(*tasks)

asyncio.run(main())
