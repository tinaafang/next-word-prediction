"""Finetune GPT2 models by designing a custom loss function."""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.CRITICAL)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'


class CustomLoss(torch.nn.Module):
    """Custom loss function that minimizes loss if desired token is predicted."""
    def __init__(self):
        """The init function."""
        super(CustomLoss, self).__init__()

    def forward(self, outputs, desired_token_id, default_loss):
        """The actual loss function that returns 0 if the model prediction is desired, else return 10 times the default loss."""
        probabilities = torch.softmax(outputs[1][:, -1, :], dim=-1)
        most_probable_token_id = torch.argmax(probabilities, dim=-1)
        if most_probable_token_id == desired_token_id:
            return torch.tensor(0.0, requires_grad=True)
        else:
            return default_loss*10


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


def finetuning(gpt2, epochs, learning_rate):
    """Finetune the model using custom loss function."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    if not gpt2:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device)

    model_name = 'GPT2' if gpt2 else 'GPT2 Large'
    print(f"{model_name} results:")

    # get token ids of label and " stretches"
    label_ids = tokenizer(["Ludwig the cat stretches"], return_tensors='pt').input_ids.to(device)
    stretches_id = tokenizer([" stretches"], return_tensors="pt").input_ids.to(device)

    # evaluate the model before finetuning
    evaluate(before=True, model=model, tokenizer=tokenizer)

    # training loop
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = CustomLoss()
    for epoch in range(epochs):
        outputs = model(label_ids, labels=label_ids)
        default_loss = outputs.loss
        loss = loss_fn(outputs, stretches_id, default_loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()

    # evaluate after finetuning
    evaluate(before=False, model=model, tokenizer=tokenizer)


if __name__ == '__main__':
    finetuning(gpt2=True, epochs=10, learning_rate=0.05)
    print("---------------------------------------------------")
    finetuning(gpt2=False, epochs=10, learning_rate=0.05)
