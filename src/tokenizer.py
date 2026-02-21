from src.logger import logger
class Tokenizer:
    def __init__(self, tokenizer_object: object) -> None:
        self.tokenizer_object = tokenizer_object
    
    def tokenize(self, text: str, return_tensors: str = "pt"):
        logger.info(f"Tokenizing text: {text} with return_tensors={return_tensors}", context="Tokenizer.tokenize")
        try:
            tokens = self.tokenizer_object(text, return_tensors=return_tensors)
            logger.info("Tokenization successful", context="Tokenizer.tokenize")
            return tokens
        except Exception as e:
            logger.error(f"Tokenization failed: {str(e)}", context="Tokenizer.tokenize")
            raise