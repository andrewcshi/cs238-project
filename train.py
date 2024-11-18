"""
TODOs:
  - fix teacher-student model parameters
  - try different hyperparams
  - try different datasets
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
import math
import random
from tqdm import tqdm

from model import Teacher, Student, TeacherStudentNetwork, collate_fn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt, src_mask, tgt_mask in tqdm(dataloader, desc="training", leave=False):
        src = src.to(device)
        tgt_input = tgt[:, :-1].to(device)
        tgt_output = tgt[:, 1:].to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask[:, :-1].to(device)

        optimizer.zero_grad()
        output = model(
            src, tgt_input,
            src_pad_mask=(src_mask == 0), tgt_pad_mask=(tgt_mask == 0)
        )
        
        # reshape outputs and targets for loss computation
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        loss = criterion(output, tgt_output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    # teacher hyperparameters
    embedding_dim_teacher = 256
    nhead_teacher = 8
    hidden_dim_teacher = 512
    teacher_num_layers = 6
    dropout_teacher = 0.1

    # student hyperparameters
    embedding_dim_student = 128
    nhead_student = 4
    hidden_dim_student = 256
    student_num_layers = 2
    dropout_student = 0.1

    max_seq_length = 128
    batch_size = 32  # change this if you don't have a GPU
    num_epochs = 5
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset and tokenizer
    dataset = load_dataset('gsm8k', 'main')

    print('assuming gpt2 tokenizer for now...')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)
    
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_seq_length)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_seq_length)
    )

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    ######################
    # Train Teacher Model
    ######################
    print("now starting training for teacher model...")
    teacher = Teacher(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim_teacher,
        nhead=nhead_teacher,
        hidden_dim=hidden_dim_teacher,
        num_layers=teacher_num_layers,
        dropout=dropout_teacher
    ).to(device)

    print(f"number of parameters in teacher model: {count_parameters(teacher)}")
    
    optimizer_teacher = torch.optim.Adam(teacher.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train_model(teacher, train_loader, optimizer_teacher, criterion, device)
        print(f"teacher model | epoch {epoch+1}/{num_epochs}, loss: {train_loss:.4f}")

    teacher_weights_path = "models/teacher.pth"
    torch.save(teacher.state_dict(), teacher_weights_path)
    print(f"teacher model saved to {teacher_weights_path}")

    ######################
    # Train Student Model
    ######################
    print("now starting training for student model...")
    student = Student(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim_student,  # smaller embedding dimension
        nhead=nhead_student,                  # fewer attention heads
        hidden_dim=hidden_dim_student,        # smaller hidden dimension
        num_layers=student_num_layers,        # fewer transformer layers
        dropout=dropout_student
    ).to(device)
    
    print(f"number of parameters in student model: {count_parameters(student)}")

    optimizer_student = torch.optim.Adam(student.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        student.train()
        total_loss = 0
        for src, tgt, src_mask, tgt_mask in tqdm(train_loader, desc=f"student epoch {epoch+1}", leave=False):
            src = src.to(device)
            tgt_input = tgt[:, :-1].to(device)
            tgt_output = tgt[:, 1:].to(device)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask[:, :-1].to(device)

            with torch.no_grad():
                teacher_memory = teacher.transformer.encoder(
                    teacher.pos_encoder(teacher.embedding(src) * math.sqrt(teacher.embedding_dim)).transpose(0,1),
                    src_key_padding_mask=(src_mask == 0)
                )

            optimizer_student.zero_grad()
            output = student(
                src, tgt_input, teacher_output=teacher_memory,
                src_pad_mask=(src_mask == 0), tgt_pad_mask=(tgt_mask == 0)
            )
            
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            optimizer_student.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"student model | epoch {epoch+1}/{num_epochs}, loss: {avg_loss:.4f}")

    student_weights_path = "models/student.pth"
    torch.save(student.state_dict(), student_weights_path)
    print(f"student model saved to {student_weights_path}")

    ###############################
    # Train Teacher-Student Network
    ###############################
    print("now starting training for teacher-student network...")
    t_s_network = TeacherStudentNetwork(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim_teacher,  # must match teacher's embedding_dim
        nhead=nhead_teacher,                  # must match teacher's nhead
        hidden_dim=hidden_dim_teacher,        # must match teacher's hidden_dim
        teacher_num_layers=teacher_num_layers,
        student_num_layers=student_num_layers,
        teacher_weights_path=teacher_weights_path,
        dropout=dropout_teacher  # must match teacher's dropout
    ).to(device)

    print(f"number of parameters in teacher-student network: {count_parameters(t_s_network)}")
    
    optimizer_ts = torch.optim.Adam(t_s_network.student.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        t_s_network.student.train()
        total_loss = 0
        for src, tgt, src_mask, tgt_mask in tqdm(train_loader, desc=f"teacher-student network epoch {epoch+1}", leave=False):
            src = src.to(device)
            tgt_input = tgt[:, :-1].to(device)
            tgt_output = tgt[:, 1:].to(device)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask[:, :-1].to(device)

            optimizer_ts.zero_grad()
            output = t_s_network(
                src, tgt_input,
                src_pad_mask=(src_mask == 0), tgt_pad_mask=(tgt_mask == 0)
            )
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            optimizer_ts.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"teacher-student network | epoch {epoch+1}/{num_epochs}, loss: {avg_loss:.4f}")

    ts_weights_path = "models/teacher_student.pth"
    torch.save(t_s_network.state_dict(), ts_weights_path)
    print(f"teacher-student network saved to {ts_weights_path}")

if __name__ == "__main__":
    main()