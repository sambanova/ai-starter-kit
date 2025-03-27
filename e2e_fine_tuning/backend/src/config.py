from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    sambastudio_host_name: str
    sambastudio_access_key: str
    sambastudio_tenant_name: str
    redis_url: str

    class ConfigDict:
        env_file = '.env'


settings = Settings()
