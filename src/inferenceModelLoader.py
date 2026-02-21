from src.logger import logger
from typing import TypeVar

T = TypeVar('T')

class InferenceModelLoader:
    def __init__(self,model: object, path: str) -> None:
        self.model = model
        self.path = path
        

    def load_model(self) -> object:
        try:
            model = self.model.from_pretrained(self.path)
            logger.info(f"Model loaded successfully from {self.path}", context="InferenceModelLoader.load_model")
            model.eval()
            logger.info("Model set to evaluation mode", context="InferenceModelLoader.load_model")
            return model
        except Exception as ex:
            logger.error(f"Failed to load model from {self.path}: {str(ex)}", context="InferenceModelLoader.load_model")
            raise