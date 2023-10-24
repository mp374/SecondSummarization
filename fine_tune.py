import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup
from read_data_set import *

stories, summaries = get_list_all_stories()


# Load the T5 model and tokenizer
model_name = "t5-small"  # You can choose other variants like 't5-base' or 't5-large'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Prepare your custom summarization dataset (source_text and target_text)
source_texts = stories  # List of source texts (input for the summarization task)
target_texts = summaries  # List of corresponding target (summary) texts

# Tokenize the source and target texts
input_ids = tokenizer.batch_encode_plus(source_texts, padding='max_length', max_length=512, return_tensors='pt', truncation=True)
labels = tokenizer.batch_encode_plus(target_texts, padding='max_length', max_length=150, return_tensors='pt', truncation=True)

# Prepare DataLoader for training
train_dataset = TensorDataset(input_ids['input_ids'], input_ids['attention_mask'], labels['input_ids'])
train_dataloader = DataLoader(train_dataset, batch_size=4)

# Define the AdamW optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_dataloader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Fine-tune the model on the summarization task
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(3):
    model.train()
    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch

        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_summarizer")
tokenizer.save_pretrained("./fine_tuned_summarizer")