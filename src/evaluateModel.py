from inferenceModelLoader import ModelLoader, Mode
from logger import logger
from transformers import GPT2Model, GPT2Tokenizer
from tokenizer import Tokenizer
from config import INFERENCE
from createAndAttachHook import CreateAndAttachHook
from detechHook import DetachHook
import torch
from torch.profiler import profile, ProfilerActivity
from memoryCalculator import MemoryCalculator
def print_summary(layer_stats):
    print(f"\n{'='*85}")
    print(f"{'Layer':<45} {'Type':<20} {'FLOPs':>10} {'Params':>10}")
    print(f"{'='*85}")

    for r in layer_stats:
        layer_short = r["layer"][-44:]  # trim long names
        print(f"{layer_short:<45} {r['type']:<20} {r['flops']:>10,} {r['params']:>10,}")

    print(f"{'='*85}")
    print(f"{'TOTAL':<45} {'':<20} {sum(r['flops'] for r in layer_stats):>10,} {sum(r['params'] for r in layer_stats):>10,}")
    print(f"{'='*85}\n")

class Evaluator:
    def __init__(self,model_path: str) -> None:
        self.model_path = model_path
    
    def evaluate(self,text: str,mode: str) -> None:
        try:
            model_loader = ModelLoader(GPT2Model, self.model_path)
            model = model_loader.load_model()
            logger.info("Model loaded", context="Evaluator.evaluate")
            mode = Mode(mode)
            model = mode.set_mode(model)
            logger.info("Model in evaluation mode", context="Evaluator.evaluate")
            tokenizer = Tokenizer(GPT2Tokenizer, self.model_path)
            tokens = tokenizer.tokenize(text, return_tensors="pt")
            # logger.info(f"Tokens: {tokens}", context="Evaluator.evaluate")
            logger.info("Evaluation started", context="Evaluator.evaluate")
            hookObj = CreateAndAttachHook(model)
            hooks, layer_stats = hookObj.attach_hooks()
            with profile(
                activities=[ProfilerActivity.CPU],  # or ProfilerActivity.CUDA for GPU
                with_flops=True                     # tells profiler to count actual FLOPs
            ) as prof:
                with torch.no_grad():
                    outputs = model(**tokens)
                    logger.info("Model forward pass completed", context="Evaluator.evaluate")
            detach_hook = DetachHook(hooks)
            detach_hook.detach_hooks()
            logger.info("Evaluation completed", context="Evaluator.evaluate")
            print_summary(layer_stats)
            # logger.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20), context="Evaluator.evaluate")
            total_flops = sum(event.flops for event in prof.key_averages())
            total_kflops = total_flops / 1000
            logger.info(f"Total  FLOPs: {total_flops:,}", context="Evaluator.evaluate")
            memory_calculator = MemoryCalculator(model, tokens)
            memory_calculator.print_memory_summary(layer_stats)
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}", context="Evaluator.evaluate")
            raise

if __name__ == "__main__":
    evaluator = Evaluator("gpt2")
    evaluator.evaluate("Hello, how am I doing? I have been doing good. What's going on?", INFERENCE)
