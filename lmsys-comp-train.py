#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install  -q kaggle kagglehub
# !kaggle datasets download -d raedmughaus/bitsandbytes-0-42-0-py3-none-any-whl
# # !kaggle datasets download -d jatinsinghsagoi/lmsys-200k-trainig-data
# !kaggle competitions download -c lmsys-chatbot-arena


# In[2]:


# !unzip /teamspace/studios/this_studio/bitsandbytes-0-42-0-py3-none-any-whl.zip
# !unzip /teamspace/studios/this_studio/lmsys-chatbot-arena


# In[3]:


# import kagglehub

# kagglehub.model_download("jatinsinghsagoi/peft-wheel/pyTorch/version1")
# kagglehub.model_download("metaresearch/llama-3/transformers/8b-hf")


# In[4]:


# !pip install -U bitsandbytes-0.42.0-py3-none-any.whl -qq
# !pip install -U peft-wheel/pyTorch/version1/1/peft-0.10.0-py3-none-any.whl -qq
# # !pip install -q trl accelerate
# !pip install -q -U git+https://github.com/huggingface/accelerate.git
# # !pip install -i https://pypi.org/simple/ bitsandbytes
# !pip install -q transformers accelerate peft datasets bitsandbytes torch


# In[5]:


import pandas as pd
from datasets import Dataset
import wandb
import torch
from transformers import AutoTokenizer, LlamaModel, LlamaForSequenceClassification, BitsAndBytesConfig
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch.nn as nn
import transformers
from sklearn.metrics import log_loss
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
import torch.nn.functional as F
from tqdm import tqdm


# In[6]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False


# In[7]:


if DEBUG:
    df = pd.read_csv("train.csv",nrows=10)
else:
    df = pd.read_csv("train.csv")
print(df.shape)
df.head()


# In[8]:


def strip(row):
    return row.strip('[]')
df['prompt'] = df['prompt'].apply(strip)
df['response_a'] = df['response_a'].apply(strip)
df['response_b'] = df['response_b'].apply(strip)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.head()


# In[9]:


config = {
    "model_name": "llama-3/transformers/8b-hf/1",
    "max_length": 1284,
    "batch_size": 8,
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.10,
    "lr": 2e-5,
    "num_epochs": 1,
    "num_warmup_steps": 100
}


# In[10]:


# from kaggle_secrets import UserSecretsClient

# user_secrets = UserSecretsClient()

# my_secret = user_secrets.get_secret("wandb") 

wandb.login(key="3f6cbde2c69fb457986fc88500866ae2893e2c87")
wandb.init(project="LMSYS",name = "LMSYS train data", config=config)


# In[11]:


df['labels'] = df[['winner_model_a','winner_model_b','winner_tie']].values.tolist()
df['prompt'] = 'User prompt: ' + df['prompt'] +  '\n\nModel A :\n' + df['response_a'] +'\n\n-----------\n\nModel B:\n'  + df['response_b']


# In[12]:


data = Dataset.from_pandas(df[['prompt','labels']])


# In[13]:


split_data = data.train_test_split(test_size=.2)
train_data = split_data['train']
eval_data = split_data['test']


# In[14]:


torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


# In[15]:


MODEL_NAME = config.get('model_name')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = LlamaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    device_map="auto",
    quantization_config=bnb_config
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token


# In[16]:


lora_config = LoraConfig(
    r=config.get('r'),
    lora_alpha=config.get('lora_alpha'),
    lora_dropout=0.10,
    bias='none',
    task_type=TaskType.SEQ_CLS,
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
)

model = get_peft_model(model, lora_config)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1


# In[17]:


def tokenize_function(examples):
    return tokenizer(examples['prompt'], padding="max_length", truncation=True, max_length = config.get('max_length'))


# In[18]:


train_dataset = train_data.map(tokenize_function, batched=True)
eval_dataset = eval_data.map(tokenize_function, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


# In[ ]:





# In[19]:


model.to(device)
optimizer = AdamW(model.parameters(), config.get('lr'))
num_epochs = config.get('num_epochs')

train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size'), shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=config.get('batch_size'), shuffle=False)

num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=config.get('num_warmup_steps'), num_training_steps=num_training_steps
)


# In[20]:


def compute_loss(logits, labels):
    logits = logits.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
#     logits = softmax(logits, axis=1)
    logits = F.softmax(torch.tensor(logits), dim=1).numpy()
    log_loss_value = log_loss(labels, logits)
    log_loss_value_tensor = torch.tensor(log_loss_value, requires_grad=True)
    
    return log_loss_value_tensor


# In[21]:


def train_epoch(model, dataloader, optimizer, device, lr_scheduler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}")
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = compute_loss(logits, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    wandb.log({"Train Loss": avg_loss, "Epoch": epoch})
    return avg_loss

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Eval Epoch {epoch+1}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = compute_loss(logits, labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    wandb.log({"Val Loss": avg_loss, "Epoch": epoch})
    return avg_loss


# In[22]:


import warnings
warnings.filterwarnings("ignore")


# In[23]:


# Early stopping parameters
patience = 1
best_eval_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train_epoch(model, train_loader, optimizer, device, lr_scheduler)
    print(f"Train loss: {train_loss:.4f}")
    eval_loss = evaluate(model, eval_loader, device)
    print(f"Eval loss: {eval_loss:.4f}")
    
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        epochs_without_improvement = 0
        model.save_pretrained("best-trained-lora")        
    else:
        epochs_without_improvement += 1
        
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break


# In[24]:


torch.save(model.state_dict(), 'final_model.pt')
model.save_pretrained("trained-model-lora")


# In[25]:


artifact = wandb.Artifact('lora_model', type='model')
artifact.add_file("/teamspace/studios/this_studio/trained-model-lora/adapter_config.json")
artifact.add_file("/teamspace/studios/this_studio/trained-model-lora/adapter_model.safetensors")
artifact.add_file("/teamspace/studios/this_studio/trained-model-lora/README.md")

wandb.log_artifact(artifact)


# In[26]:


# !pip install dagshub


# In[1]:


from dagshub import get_repo_bucket_client
s3 = get_repo_bucket_client("JatinSingh28/LMSYS-lora-finetune-for-llm-ranking")


# In[8]:


import os


# In[9]:


base_path = "trained-model-lora"
for path in os.listdir(base_path):
    try:
        s3.upload_file(
                Filename=os.path.join(base_path,path),
                Bucket="LMSYS-lora-finetune-for-llm-ranking",
                Key="lora/"+path
            )
    except Exception as e:
        print(f"Couldn't upload model to dagshub. Error: {e}")


# In[10]:


base_path = "best-trained-lora"
for path in os.listdir(base_path):
    try:
        s3.upload_file(
                Filename=os.path.join(base_path,path),
                Bucket="LMSYS-lora-finetune-for-llm-ranking",
                Key="best-lora/"+path
            )
    except Exception as e:
        print(f"Couldn't upload model to dagshub. Error: {e}")


# In[6]:


# s3.download_file(
#     Bucket="LMSYS-lora-finetune-for-llm-ranking",  
#     Key="lora/README.md",  
#     Filename="downloads/README.md",  
# )
# s3.download_file(
#     Bucket="LMSYS-lora-finetune-for-llm-ranking",  
#     Key="lora/adapter_config.json",  
#     Filename="downloads/adapter_config.json",  
# )
# s3.download_file(
#     Bucket="LMSYS-lora-finetune-for-llm-ranking",  
#     Key="lora/adapter_model.safetensors",  
#     Filename="downloads/adapter_model.safetensors",  
# )


# In[30]:


wandb.finish()

