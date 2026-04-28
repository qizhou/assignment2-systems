from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
import timeit
import torch
import model_params

d_model, d_ff, num_layers, num_heads = model_params.get_model_parameters(model_params.MODEL_LARGE)
vocab_size = 10000
batch_size = 4
context_length = 256
w = 5 # warmup step
m = 10 # number of measurement

torch.set_default_device("cuda")
model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff)
dtype = torch.bfloat16

forward_used_time = []
backward_used_time = []
print(model.get_num_params())
with torch.autocast(device_type="cuda", dtype=dtype):
    for i in range(w+m):
        batch = torch.randint(vocab_size, (batch_size, context_length+1))
        inputs = batch[:, :context_length]
        targets = batch[:, 1:context_length+1]

        # forward
        start = timeit.default_timer()
        logits = model(inputs)
        torch.cuda.synchronize()
        if i >= w:
            forward_used_time.append(timeit.default_timer() - start)

        # backward
        start = timeit.default_timer()
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        torch.cuda.synchronize()
        if i >= w:
            backward_used_time.append(timeit.default_timer() - start)

    print(torch.mean(torch.tensor(forward_used_time)), torch.std(torch.tensor(forward_used_time)))
    print(torch.mean(torch.tensor(backward_used_time)), torch.std(torch.tensor(backward_used_time)))
