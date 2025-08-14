class APIKeyManager:
    async def create_api_key(self, *args, **kwargs):
        return {"api_key": "test-api-key"}
    async def validate_api_key(self, api_key):
        return True
    async def revoke_api_key(self, api_key):
        return True 