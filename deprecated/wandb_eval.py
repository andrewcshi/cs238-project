import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
from model import Teacher, Student, TeacherStudentNetwork
import torch.nn as nn

# Evaluation Function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt, src_mask, tgt_mask in dataloader:
            src = src.to(device)
            tgt_input = tgt[:, :-1].to(device)
            tgt_output = tgt[:, 1:].to(device)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask[:, :-1].to(device)

            output = model(src, tgt_input, src_pad_mask=(src_mask == 0), tgt_pad_mask=(tgt_mask == 0))
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == "__main__":
    # Hyperparameters
    embedding_dim = 256
    nhead = 8
    hidden_dim = 512
    teacher_num_layers = 6
    student_num_layers = 3
    dropout = 0.1
    max_seq_length = 128
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load GSM8K Dataset
    dataset = load_dataset('gsm8k', 'main')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # Data Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_seq_length)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_seq_length)
    )

    # Initialize Teacher Model
    teacher = Teacher(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        nhead=nhead,
        hidden_dim=hidden_dim,
        num_layers=teacher_num_layers,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Evaluate Models
    print("Evaluating Models...")
    teacher_loss = evaluate_model(teacher, test_loader, criterion, device)
    student_loss = evaluate_model(student, test_loader, criterion, device)

    print(f"Teacher Model Test Loss: {teacher_loss:.4f}")
    print(f"Student Model Test Loss: {student_loss:.4f}")

    # init teacher-student network
    t_s_network = TeacherStudentNetwork(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        nhead=nhead,
        hidden_dim=hidden_dim,
        teacher_num_layers=teacher_num_layers,
        student_num_layers=student_num_layers,
        teacher_weights_path=teacher_weights_path,
        dropout=dropout
    ).to(device)

    # eval teacher-student network
    t_s_network.student.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt, src_mask, tgt_mask in test_loader:
            src = src.to(device)
            tgt_input = tgt[:, :-1].to(device)
            tgt_output = tgt[:, 1:].to(device)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask[:, :-1].to(device)

            output = t_s_network(
                src, tgt_input,
                src_pad_mask=(src_mask == 0), tgt_pad_mask=(tgt_mask == 0)
            )
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()
    ts_loss = total_loss / len(test_loader)
    print(f"teacher-student network test loss: {ts_loss:.4f}")

    # summarize evaluation results
    print("\nsummary of model performance:")
    print(f"teacher model test loss: {teacher_loss:.4f}")
    print(f"student model test loss: {student_loss:.4f}")
    print(f"teacher-student network test loss: {ts_loss:.4f}")
