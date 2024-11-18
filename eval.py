"""
this doesn't really do anything interesting yet

TODOs:
  - set up eval graphs to evaluate hyperparams
  - set up wandb logging for sampling from models
  - find/run tests on different datasets
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
import math

from model import Teacher, Student, TeacherStudentNetwork, collate_fn

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

def main():
    # hyperparameters (should match training)
    embedding_dim_teacher = 256
    nhead_teacher = 8
    hidden_dim_teacher = 512
    teacher_num_layers = 6
    dropout_teacher = 0.1

    embedding_dim_student = 128
    nhead_student = 4
    hidden_dim_student = 256
    student_num_layers = 2
    dropout_student = 0.1

    max_seq_length = 128
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = load_dataset('gsm8k', 'main')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)
    
    test_dataset = dataset['test']

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_seq_length)
    )

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # load teacher model
    teacher = Teacher(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim_teacher,
        nhead=nhead_teacher,
        hidden_dim=hidden_dim_teacher,
        num_layers=teacher_num_layers,
        dropout=dropout_teacher
    ).to(device)
    teacher_weights_path = "models/teacher.pth"
    teacher.load_state_dict(torch.load(teacher_weights_path))
    teacher.eval()
    print(f"loaded teacher model from {teacher_weights_path}")

    # load student model
    student = Student(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim_student,
        nhead=nhead_student,
        hidden_dim=hidden_dim_student,
        num_layers=student_num_layers,
        dropout=dropout_student
    ).to(device)
    student_weights_path = "models/student.pth"
    student.load_state_dict(torch.load(student_weights_path))
    student.eval()
    print(f"loaded student model from {student_weights_path}")

    # load teacher-student network
    ts_network = TeacherStudentNetwork(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim_teacher,
        nhead=nhead_teacher,
        hidden_dim=hidden_dim_teacher,
        teacher_num_layers=teacher_num_layers,
        student_num_layers=student_num_layers,
        teacher_weights_path=teacher_weights_path,
        dropout=dropout_teacher
    ).to(device)
    ts_weights_path = "models/teacher_student.pth"
    ts_network.load_state_dict(torch.load(ts_weights_path))
    ts_network.eval()
    print(f"loaded teacher-student network from {ts_weights_path}")

    # evaluate models
    print("evaluating teacher model...")
    teacher_loss = evaluate_model(teacher, test_loader, criterion, device)
    print(f"teacher model test loss: {teacher_loss:.4f}")

    print("evaluating student model...")
    student_loss = evaluate_model(student, test_loader, criterion, device)
    print(f"student model test loss: {student_loss:.4f}")

    print("evaluating teacher-student network...")
    ts_loss = evaluate_model(ts_network, test_loader, criterion, device)
    print(f"teacher-student network test loss: {ts_loss:.4f}")

    # Summary
    print("\nsummary of evals:")
    print(f"teacher model test loss: {teacher_loss:.4f}")
    print(f"student model test loss: {student_loss:.4f}")
    print(f"teacher-student network test loss: {ts_loss:.4f}")

if __name__ == "__main__":
    main()