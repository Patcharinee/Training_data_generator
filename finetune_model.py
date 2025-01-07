import logging
import transformers
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_dataset


logger = logging.getLogger(name)
global_config = None

### Load the training dataset that you've prepared ###
# using file from Huggingface (hf)
#dataset_path = "dataset_path"
#use_hf = True  #flag to specify whether we load training data file from Huggingface or not


# or use the following code to load local training data (not using Huggingface)
dataset_name = "lamini_docs.jsonl"
dataset_path = f"/content/{dataset_name}"
use_hf = False

### Set up the model, training config, tokenizer, split data into train dataset and test dataset ###
model_name = "EleutherAI/pythia-70m"

training_config = {
    "model": {
        "pretrained_name": model_name,
        "max_length" : 2048
    },
    "datasets": {
        "use_hf": use_hf,
        "path": dataset_path
    },
    "verbose": True
}

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
train_dataset, test_dataset = tokenize_and_split_data(training_config, tokenizer)

print(train_dataset)
print(test_dataset)

### Load the base model (the pretrained model one (not finetuned)) ###
base_model = AutoModelForCausalLM.from_pretrained(model_name)

### Put base model on the device (GPU or CPU device that we have)
#Pytorch code to count #of GPU that we have and use GPU if there is any (o.w. use CPU)
device_count = torch.cuda.device_count()
if device_count > 0:
    logger.debug("Select GPU device")
    device = torch.device("cuda")
else:
    logger.debug("Select CPU device")
    device = torch.device("cpu")
# put the model on the device (GPU or CPU)
base_model.to(device) 


### Define function to carry out inference ### 
# max_input_tokens = max tokens to be input to the model
# max_output_tokens = max tokens to be generated from the model
def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
    # Tokenize
    input_ids = tokenizer.encode(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens
    )

    # Generate
    device = model.device
    generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device), # put the tokens of dataset to the same device as the model
    max_length=max_output_tokens
    )

    # Decode
    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

    # Strip the prompt
    generated_text_answer = generated_text_with_prompt[0][len(text):]

    return generated_text_answer
  
### Setup training/finetuning ###

# max number of training steps == max number of batches of training data that we will run on the model
# step = batch of training data 
max_steps = 3 

trained_model_name = f"lamini_docs_{max_steps}_steps" #we can also put timestamp here to differentiate between different model trainings
output_dir = trained_model_name

training_args = TrainingArguments(
    # Learning rate
    learning_rate=1.0e-5,

    # Number of training epochs
    num_train_epochs=1,

    # Max steps to train for (each step is a batch of data)
    # Overrides num_train_epochs, if not -1
    max_steps=max_steps,

    # Batch size for training
    per_device_train_batch_size=1,

    # Directory to save model checkpoints
    output_dir=output_dir,

    # Other arguments
    overwrite_output_dir=False, # Overwrite the content of the output directory
    disable_tqdm=False, # Disable progress bars
    eval_steps=120, # Number of update steps between two evaluations
    save_steps=120, # After # steps model is saved
    warmup_steps=1, # Number of warmup steps for learning rate scheduler
    per_device_eval_batch_size=1, # Batch size for evaluation
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=1,
    optim="adafactor",
    gradient_accumulation_steps = 4,
    gradient_checkpointing=False,

    # Parameters for early stopping
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="eval_loss",
    greater_is_better=False
    )

# Just to calculate the model FLOPS and memory footprint
model_flops = (
    base_model.floating_point_ops(
        {
            "input_ids": torch.zeros(
                (1, training_config["model"]["max_length"])
                )
        }
    )
  * training_args.gradient_accumulation_steps
)

print(base_model)
print("Memory footprint", base_model.get_memory_footprint() / 1e9, "GB")
print("Flops", model_flops / 1e9, "GFLOPs")

# Trainer class to print out information during the model training
trainer = Trainer(
    model=base_model,
    model_flops=model_flops,
    total_steps=max_steps,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

### TRAIN/FINETUNE THE MODEL ###
training_output = trainer.train()

### Save the trained model locally ###
save_dir = f'{output_dir}/final'

trainer.save_model(save_dir)
print("Saved model to:", save_dir)

### Load the trained model from local directory ###
finetuned_slightly_model = AutoModelForCausalLM.from_pretrained(save_dir, local_files_only=True) 
# local_files_only = True means get the model from local directory not from Huggingface

# put the trained model on the device (i.e., CPU in our example)
finetuned_slightly_model.to(device) 

### Try to run the trained model on a test question to see whether it performs better ###
test_question = test_dataset[0]['question']
print("Question input (test):", test_question)

print("Finetuned slightly model's answer: ")
print(inference(test_question, finetuned_slightly_model, tokenizer))

test_answer = test_dataset[0]['answer']
print("Target answer output (test):", test_answer)