from logger import logger
from typing import TypeVar
T = TypeVar('T')
from config import INFERENCE
class Mode:
    def __init__(self, mode: str) -> None:
        self.mode = mode
    
    def set_mode(self,model) -> object:
        if self.mode == INFERENCE:
            model.eval()
        return model

class ModelLoader:
    def __init__(self,model: object, path: str) -> None:
        self.model = model
        self.path = path
        

    def load_model(self) -> object:
        try:
            model = self.model.from_pretrained(self.path)
            return model
        except Exception as ex:
            logger.error(f"Failed to load model from {self.path}: {str(ex)}", context="InferenceModelLoader.load_model")
            raise