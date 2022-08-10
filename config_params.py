from pydantic import BaseModel
import yaml


class Training(BaseModel):
    seq_length: int = 300
    batch_size: int = 32
    learning_rate: float = 0.004
    neurons: int = 128
    layers: list = []
    epochs: int = 100
    patience: int = 10


class Data(BaseModel):
    composer: list
    source: str


class Composition(BaseModel):
    temperature: float = 1.0
    num_predictions: int = 2


class Settings(BaseModel):
    training: Training
    data: Data
    composition: Composition


with open("config.yml") as f:
    data = yaml.safe_load(f)

settings = Settings(**data)