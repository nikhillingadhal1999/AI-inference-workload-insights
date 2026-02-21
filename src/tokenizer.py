from logger import logger
class Tokenizer:
    def __init__(self, tokenizer_object: object, model_path: str) -> None:
        self.tokenizer_object = tokenizer_object
        self.model_path = model_path
    
    def tokenize(self, text: str, return_tensors: str = "pt"):
        try:
            tokens = self.tokenizer_object.from_pretrained(self.model_path)(text, return_tensors=return_tensors)
            logger.info("Tokenization successful", context="Tokenizer.tokenize")
            return tokens
        except Exception as e:
            logger.error(f"Tokenization failed: {str(e)}", context="Tokenizer.tokenize")
            raise