from apis.api_client import APIClient
import asyncio


class Map4dSDKClient:
    
    def __init__(
            self,
            map_url = 'https://api.map4d.vn/sdk',
            api_key = 'eaecf61a20ca4edbe1024d31d6595b71'
        ):
        self.MAP_URL = map_url
        self.API_KEY = api_key
        self.API_PATH = {
            'route': '/route',
            'geocode': '/v2/geocode',
            'search_nearby': '/place/nearby-search',
        }
    
    def api_route(
            self,
            start_lat,
            start_lon,
            end_lat,
            end_lon
        ):
        self.api_base = self.API_PATH['route']
        self.full_url = self.MAP_URL + self.api_base
        self.API_CLIENT = APIClient(self.full_url)
        params = {
            'key': self.API_KEY,
            'origin': str(start_lat) + ',' + str(start_lon),
            'destination': str(end_lat) + ',' + str(end_lon),
            'mode': 'foot',
            'language': 'vi',
            'weighting': 0
        }
        data = asyncio.run(self.API_CLIENT.fetch_data(params))
        return data

    def api_geo_coding(self, lat, lon):
        self.api_base = self.API_PATH['geocode']
        self.full_url = self.MAP_URL + self.api_base
        self.API_CLIENT = APIClient(self.full_url)
        params = {
            'key': self.API_KEY,
            'location': str(lat) + ',' + str(lon)
        }
        data = asyncio.run(self.API_CLIENT.fetch_data(params))
        return data

    def api_search_nearby(
        self,
        lat,
        lon,
        radius=10,
        place_type='point'
    ):
        self.api_base = self.API_PATH['search_nearby']
        self.full_url = self.MAP_URL + self.api_base
        self.API_CLIENT = APIClient(self.full_url)
        params = {
            "key": self.API_KEY,
            "location": f"{lat},{lon}",
            "radius": radius,
            "types": place_type,
        }
        data = asyncio.run(self.API_CLIENT.fetch_data(params))
        return data
