from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch, wandb
from datasets import load_dataset
from trl import SFTTrainer

hf_token = os.getenv("HUGGINGFACE_TOKEN")
wandb_token = os.getenv("wandb")

wandb.login(key = wandb_token)
run = wandb.init(
    project='Fine tuning mistral 7B', 
    job_type="training", 
    anonymous="allow"
)

base_model = "mistralai/Mistral-7B-v0.1"

dataset_name = "mlabonne/guanaco-llama2-1k"
new_model = "mistral-7b-guanco-full"


dataset = load_dataset(dataset_name, split="train")

dataset['text'][100]

bnb_config = BitsAndBytesConfig(  
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)

model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
)
model.config.use_cache = False # silence the warnings
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
)
model = get_peft_model(model, peft_config)

training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()

trainer.model.save_pretrained(new_model)
wandb.finish()
model.config.use_cache = True

logging.set_verbosity(logging.CRITICAL)

prompt = "How do I find true love?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

prompt = "What is Datacamp Career track?"
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

trainer.model.save_pretrained(new_model)
trainer.model.push_to_hub(new_model, use_temp_dir=False)


from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline 
from peft import PeftModel
import torch

base_model_reload = AutoModelForCausalLM.from_pretrained(
        base_model,
        return_dict=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model_reload, new_model)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer = tokenizer, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

prompt = "How become a DataCamp certified data professional"

sequences = pipe(
    f"<s>[INST] {prompt} [/INST]",
    do_sample=True,
    max_new_tokens=100, 
    temperature=0.7, 
    top_k=50, 
    top_p=0.95,
    num_return_sequences=1,
)
print(sequences[0]['generated_text'])

model = model.merge_and_unload()

model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)