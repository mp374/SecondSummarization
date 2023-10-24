import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the fine-tuned summarizer and tokenizer
model_path = "fine_tuned_summarizer"  # Path to the saved fine-tuned model
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Set device to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Text to summarize
text_to_summarize = "I cannot thank Ward 12 NSECH for their care and commitment to my dear relative who was admitted " \
                    "on the 23rd December then again on the 6th January, they were brilliant in every aspect, " \
                    "they treat her with such care and compassion, the staff work so hard yet always make time if you " \
                    "have a concern, this ward is run brilliantly, clearly a happy team, thank you Ward 12 you are " \
                    "wondeful "

# Tokenize the input text and generate summary
input_ids = tokenizer.encode("summarize: " + text_to_summarize, return_tensors="pt", max_length=512, truncation=True).to(device)
summary_ids = model.generate(input_ids, max_length=5, num_beams=4, early_stopping=True)

# Decode the summary tokens and print the generated summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Generated Summary:", summary)
