# Install the Transformers library

# Import necessary modules
from transformers import *
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the T5 model and tokenizer
from transformers.models.swiftformer.convert_swiftformer_original_to_hf import device

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Define the input text and the summary length
text = "The University of Plymouth is a public research university based predominantly in Plymouth, England, " \
       "where the main campus is located, but the university has campuses and affiliated colleges across South West " \
       "England. With 18,410 students, it is the 57th largest in the United Kingdom by total number of students. "
max_length = 20

# Preprocess the text and encode it as input for the model
input_text = "summarize: " + text
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

# Generate a summary
summary = model.generate(input_ids, max_length=max_length)

# Decode the summary
summary_text = tokenizer.decode(summary[0], skip_special_tokens=True)
print(summary_text)