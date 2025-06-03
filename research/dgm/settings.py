from pydantic_settings import BaseSettings

class DGMSettings(BaseSettings):
    # Add DGM specific settings here
    param_dgm_example: str = "default_dgm_value"

    class Config:
        env_prefix = "DGM_"
