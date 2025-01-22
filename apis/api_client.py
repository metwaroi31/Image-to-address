import aiohttp


class APIClient:
    
    def __init__(self,
                 full_url):
        self.FULL_URL = full_url
    
    async def fetch_data(self, params):
        async with aiohttp.ClientSession() as session:
            async with session.get(self.FULL_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    print(f"Error: {response.status}")
                    return None
    