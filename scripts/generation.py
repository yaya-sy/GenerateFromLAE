import torch
import torch.nn.functional as F

def greedy(model, tokenizer, encoded_source, max_length, device):
    src = torch.tensor(encoded_source).unsqueeze(0).to(device)
    generated_indexes = [tokenizer.bos_id()]
    for _ in range(max_length):
        tgt = torch.tensor(generated_indexes).unsqueeze(0).to(device)
        logits = model(src, tgt).squeeze(0)
        probs = F.softmax(logits[-1, :], dim=-1)
        next_index = torch.argmax(probs, dim=-1)
        next_index = next_index.item()
        if next_index == tokenizer.eos_id():
            return generated_indexes
        generated_indexes.append(next_index)
    return generated_indexes