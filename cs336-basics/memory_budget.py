import model_params

d_model, d_ff, num_layers, num_heads = model_params.get_model_parameters(model_params.MODEL_LARGE)
vocab_size = 10000
context_length = 256

# input embedding
embedding = vocab_size * d_model

# per layer parameters
pre_norm = d_model # gain parameter
feed_forward = 3 * d_model * d_ff
attn = d_model * d_model * 4 # d_k = d_v = d_model // num_heads

layer = 2 * pre_norm + feed_forward + attn

total = embedding + layer * num_layers + pre_norm + d_model * vocab_size
print(f"per layer: pre_norm {2 * pre_norm:,}, feed_forward {feed_forward:,}, attn {attn:,}, layer {layer:,}, total {total:,}")

# activations (can be optimized by mix precision)
act_embedding = d_model * context_length
act_attn = 4 * d_model * context_length + num_heads * context_length * context_length # second part can be optimized by flash attention
act_feed_forward = 3 * d_ff * context_length
batch = 4 # in training

print(f"inference: act_embedding {act_embedding:,}, act_attn {act_attn:,}, act_feed_forward {act_feed_forward:,}")
# for training stage, we need to store all activations from all layers and all batches to calculate gradient
print(f"training: act_embedding {act_embedding*batch:,}, act_attn {act_attn*num_layers*batch:,}, act_feed_forward {act_feed_forward*num_layers*batch:,} , total {batch*(act_embedding+(act_attn+act_feed_forward)*num_layers):,}")


