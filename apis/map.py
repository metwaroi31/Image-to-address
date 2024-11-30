import aiohttp

MAP_URL = 'http://localhost:8080/points'

async def send_post_request(json_of_point):
    json_data = {
        "osmId": "11994579471",
        "osmName": "Masjidka isbahaysiga جامع التضامن الإسلامي",
        "code": 3300,
        "fclass": "muslim",
        "geom": {
            "srid": "4326",
            "longitude": json_of_point['Longitude'],
            "latitude": json_of_point['lat']
        },
        "name": str(json_of_point['shop_name']),
        "address": str(json_of_point['address']),
        "description": str(json_of_point['product']),
        "status": "mở cửa",
        "opening_hours": "Mở lúc 7:30 Th 2 - Th 6",
        "rating": 4.6,
        "reviews": 13,
        "url": str(json_of_point['file_name']),
        "file": str(json_of_point['file_name']),
        "logo": "http://example.com/logo.png",   
        "businessType": str(json_of_point['category'])
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(MAP_URL, json=json_data) as response:
            return_data = await response.text()
            print("Status:", response.status)
            print("Response:", return_data)
            return return_data

