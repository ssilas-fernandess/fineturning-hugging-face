from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
import numpy as np
import evaluate

# 1. Mini dataset sintético: Perguntas e respostas sobre um domínio fictício
data = {
    "question": [
        "Qual o horário de funcionamento da empresa?",
        "Como solicitar reembolso?",
        "Quem é o responsável pelo setor de TI?",
        "Como acessar o sistema interno?",
        "Quais são os benefícios oferecidos?"
    ],
    "answer": [
        "A empresa funciona das 8h às 18h, de segunda a sexta.",
        "Para solicitar reembolso, preencha o formulário e envie para o financeiro.",
        "O responsável pelo setor de TI é o João Silva.",
        "O sistema interno pode ser acessado via VPN com login institucional.",
        "Oferecemos vale alimentação, plano de saúde e auxílio home-office."
    ]
}

# 2. Criação do dataset Hugging Face
raw_dataset = Dataset.from_dict(data)

# 3. Tokenização
checkpoint = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

max_input_length = 128
max_target_length = 128

def preprocess_function(example):
    inputs = ["pergunta: " + q for q in example["question"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["answer"], max_length=max_target_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

encoded_dataset = raw_dataset.map(preprocess_function, batched=True)

# 4. Métrica
metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {"rougeL": result["rougeL"].mid.fmeasure}

# 5. Configuração do treinamento com integração ao Hugging Face Hub
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=10,
    weight_decay=0.01,
    predict_with_generate=True,
    logging_dir="./logs",
    push_to_hub=True,
    hub_model_id="ssilasfernandess/fineturning-hugging-face",
    hub_strategy="every_save"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    eval_dataset=encoded_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 6. Treinamento
trainer.train()

# 7. Upload manual (opcional, caso push_to_hub esteja False)
trainer.push_to_hub()