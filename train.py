import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

from preprocess import load_fever, prepare_data, build_sentence_pool
from wiki_loader import load_all_wiki

# --------------------------
# LOAD DATA
# --------------------------
print("Loading FEVER data...")
data = load_fever("data/fever/train.jsonl")

print("Loading wiki (limited)...")
wiki = load_all_wiki("data/wiki/", max_files=1)  # 🔥 limit for speed

print("Building sentence pool...")
build_sentence_pool(wiki)  # 🔥 REQUIRED

print("Preparing dataset...")
dataset = prepare_data(data, wiki, limit=2000)  # 🔥 reduce size

print("Dataset size:", len(dataset))


# --------------------------
# MODEL
# --------------------------
print("Loading model...")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# --------------------------
# DATALOADER
# --------------------------
def collate(batch):
    claims = [x[0] for x in batch]
    evidences = [x[1] for x in batch]
    labels = torch.tensor([x[2] for x in batch])

    inputs = tokenizer(
        claims,
        evidences,                                                                                                                                              
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = labels.to(device)

    return inputs, labels


loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate
)

optimizer = AdamW(model.parameters(), lr=2e-5)


# --------------------------
# TRAIN
# --------------------------
print("Starting training...")

for epoch in range(3):
    model.train()
    total_loss = 0

    for step, (inputs, labels) in enumerate(loader):

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        # 🔥 progress log every 50 steps
        if step % 50 == 0:
            print(f"Epoch {epoch+1} | Step {step} | Loss {loss.item():.4f}")

    print(f"✅ Epoch {epoch+1} Finished | Total Loss: {total_loss:.4f}")


# --------------------------
# SAVE MODEL
# --------------------------
print("Saving model...")

model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")

print("✅ Training complete. Model saved.")