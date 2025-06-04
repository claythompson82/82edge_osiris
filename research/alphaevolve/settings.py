from pydantic_settings import BaseSettings


class AlphaEvolveSettings(BaseSettings):
    # Add AlphaEvolve specific settings here
    param_alphaevolve_example: str = "default_alphaevolve_value"

    class Config:
        env_prefix = "ALPHAEVOLVE_"
