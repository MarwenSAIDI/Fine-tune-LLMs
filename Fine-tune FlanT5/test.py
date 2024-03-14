from transformers import AutoTokenizer, T5ForConditionalGeneration
import os

##Load the tokenizer and the model for generation
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", cache_dir=os.path.join(os.getcwd(),"models-cache"), legacy=False)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", cache_dir=os.path.join(os.getcwd(),"models-cache"), device_map="auto")

##Tokenize the input text
input_text = "Give me the name in this text: Hello, my name is Marouene Saidi. Nice to meet you!"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

##Generate the responce and display it
outputs = model.generate(input_ids, max_length=100)
print(tokenizer.decode(input_ids[0]))
print(tokenizer.decode(outputs[0]))


##Memory management
del outputs
del input_ids
del input_text
del tokenizer
del model
del AutoTokenizer
del T5ForConditionalGeneration
del os