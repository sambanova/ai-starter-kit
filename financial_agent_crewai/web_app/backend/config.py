from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    redis_host: str
    redis_port: int
    secret_key: str
    algorithm: str
    access_token_expire_minutes: int

    class ConfigDict:
        env_file = '.env'


settings = Settings() # type: ignore
