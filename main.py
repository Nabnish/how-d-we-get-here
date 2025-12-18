from openai import OpenAI
client = OpenAI()

file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)

print(file.id)
