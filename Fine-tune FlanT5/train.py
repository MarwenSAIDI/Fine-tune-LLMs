from transformers import AutoTokenizer, AutoModel
import os

##Load the tokenizer and the model for generation
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", cache_dir=os.path.join(os.getcwd(),"models-cache"), legacy=False)
model = AutoModel.from_pretrained("google/flan-t5-small", cache_dir=os.path.join(os.getcwd(),"models-cache"), device_map="auto")


print(model.shared)
print("-----------------------------------")
print(model.encoder)

del tokenizer
del model
del os
del AutoTokenizer
del AutoModel