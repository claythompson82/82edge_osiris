from pydantic_settings import BaseSettings


class AZRSettings(BaseSettings):
    # Add AZR specific settings here
    param_azr_example: str = "default_azr_value"

    class Config:
        env_prefix = "AZR_"
