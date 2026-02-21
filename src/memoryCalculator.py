class MemoryCalculator:
    def __init__(self, model, input_data):
        self.model = model
        self.input_data = input_data
    
    def to_mb(self, byte_count):
        return round(byte_count / (1024 ** 2), 3)

    
    def input_activation_memory(self, token_count, hidden_size, bytes_per_param, num_layers):
        total_bytes = token_count * hidden_size * bytes_per_param * num_layers
        return self.to_mb(total_bytes)

    def calculate_memory_usage(self):
        total_params    = sum(p.numel() for p in self.model.parameters())
        total_bytes     = sum(p.numel() * p.element_size() for p in self.model.parameters())
        dtype           = next(self.model.parameters()).dtype
        bytes_per_param = next(self.model.parameters()).element_size()

        return {
            "dtype"           : dtype,
            "bytes_per_param" : bytes_per_param,
            "total_params"    : total_params,
            "total_mb"        : self.to_mb(total_bytes),
        }
    
    def print_memory_summary(self, layer_stats):
        weight_info = self.calculate_memory_usage()

        # Activation memory from hooks (actual measured values)
        total_activation_mb = self.to_mb(sum(r["output_mem"] for r in layer_stats))
        peak_activation_mb  = self.to_mb(max(r["output_mem"] for r in layer_stats))

        # Estimated activation memory from formula
        token_count = self.input_data["input_ids"].shape[1]
        hidden_size = self.model.config.hidden_size
        num_layers  = self.model.config.num_hidden_layers
        estimated_activation_mb = self.input_activation_memory(
            token_count, hidden_size, weight_info["bytes_per_param"], num_layers
        )

        total_inference_mb = weight_info["total_mb"] + peak_activation_mb

        print(f"\n{'='*55}")
        print(f"  MEMORY SUMMARY")
        print(f"{'='*55}")

        print(f"\n  -- Model Weights (fixed, independent of input) --")
        print(f"  dtype            : {weight_info['dtype']}")
        print(f"  bytes per param  : {weight_info['bytes_per_param']}  (float32=4, float16=2, int8=1)")
        print(f"  total params     : {weight_info['total_params']:,}")
        print(f"  weight memory    : {weight_info['total_mb']} MB")

        print(f"\n  -- Activation Memory (changes with input) --")
        print(f"  token count      : {token_count} tokens")
        print(f"  hidden size      : {hidden_size}")
        print(f"  num layers       : {num_layers}")
        print(f"  estimated activ  : {estimated_activation_mb} MB   (formula: tokens x hidden x layers)")
        print(f"  measured activ   : {total_activation_mb} MB   (sum of all layer outputs from hooks)")
        print(f"  peak activ       : {peak_activation_mb} MB   (largest single layer output)")

        print(f"\n  -- Total Inference Memory --")
        print(f"  weights + peak   : {total_inference_mb} MB")
        print(f"{'='*55}\n")