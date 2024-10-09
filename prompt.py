"""Finetune GPT2 and GPT2 Large by updating the non-contexualized subword embeddings of "Ludwig the cat"."""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.CRITICAL)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'


def evaluate(before, model, tokenizer):
    """Evaluate the probablity of predicting "stretches" and the loss."""
    model.eval()
    prefix = "[Before]" if before else "[After] "
    # probablity of "stretches" before finetuning
    input_ids = tokenizer(["Ludwig the cat"], return_tensors="pt").input_ids.to(device)
    outputs = model(input_ids)
    next_token_logits = outputs.logits[0, -1, :]
    next_token_probs = torch.softmax(next_token_logits, -1)
    stretches_id = tokenizer([" stretches"], return_tensors="pt").input_ids.to(device)
    print(f"{prefix} The probability of predicting \"stretches\": {next_token_probs[stretches_id.item()]:.5%}")

    # loss before finetuning
    label_ids = tokenizer(["Ludwig the cat stretches"], return_tensors='pt').input_ids.to(device)
    outputs = model(label_ids, labels=label_ids)
    print(f"{prefix} The loss: {outputs.loss.item():.5f}")


def prompt(gpt2, epochs, learning_rate):
    """Finetune the model by only updating the non-contexualized subword embeddings for "Ludwig the cat"."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    if not gpt2:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device)

    model_name = 'GPT2' if gpt2 else 'GPT2 Large'
    print(f"{model_name} results:")

    # get token ids of input and label
    input_ids = tokenizer(["Ludwig the cat"], return_tensors="pt").input_ids.to(device)
    label_ids = tokenizer(["Ludwig the cat stretches"], return_tensors='pt').input_ids.to(device)

    # evaluate the model before finetuning
    evaluate(before=True, model=model, tokenizer=tokenizer)

    # freeze the non-contextualized token embeddings for tokens that are not "Ludwig the cat"
    embedding_weights = model.transformer.wte.weight
    freeze_mask = torch.zeros(embedding_weights.size(0), dtype=torch.float32).to(embedding_weights.device)
    freeze_mask[input_ids] = 1
    embedding_weights.register_hook(lambda x: x * freeze_mask.unsqueeze(1))

    # only pass in the non-contextualized token embeddings, so the optimizer would not update other model parameters
    optimizer = torch.optim.AdamW([model.transformer.wte.weight], lr=learning_rate)

    # training loop
    for epoch in range(epochs):
        outputs = model(label_ids, labels=label_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()

    # evaluate after finetuning
    evaluate(before=False, model=model, tokenizer=tokenizer)


if __name__ == '__main__':
    prompt(gpt2=True, epochs=500, learning_rate=0.01)
    print("---------------------------------------------------")
    prompt(gpt2=False, epochs=500, learning_rate=1e-3)
