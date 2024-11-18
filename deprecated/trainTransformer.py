import random
import time
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset

# positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)  # positional encoding matrix
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# basic teacher model architecture
class Teacher(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, hidden_dim, num_layers, dropout):
        super(Teacher, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu'
        )
        
        self.generator = nn.Linear(embedding_dim, vocab_size)
        self.embedding_dim = embedding_dim
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, src_pad_mask=None, tgt_pad_mask=None):
        src_embedded = self.embedding(src) * math.sqrt(self.embedding_dim)
        src_embedded = self.pos_encoder(src_embedded)
        src_embedded = src_embedded.transpose(0, 1)  # shape: [seq_len, batch_size, embedding_dim]
        
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.embedding_dim)
        tgt_embedded = self.pos_encoder(tgt_embedded)
        tgt_embedded = tgt_embedded.transpose(0, 1)
        
        tgt_mask = self.generate_square_subsequent_mask(tgt_embedded.size(0)).to(src.device)
        
        output = self.transformer(
            src_embedded,
            tgt_embedded,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        output = self.generator(output)
        return output.transpose(0, 1)  # shape: [batch_size, tgt_seq_len, vocab_size]

# basic student model architecture
class Student(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, hidden_dim, num_layers, dropout):
        super(Student, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu'
        )
        
        self.generator = nn.Linear(embedding_dim, vocab_size)
        self.embedding_dim = embedding_dim

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, teacher_output=None, src_pad_mask=None, tgt_pad_mask=None, teachers_input_student_ratio=10):
        src_embedded = self.embedding(src) * math.sqrt(self.embedding_dim)
        src_embedded = self.pos_encoder(src_embedded)
        src_embedded = src_embedded.transpose(0, 1)  # shape: [seq_len, batch_size, embedding_dim]
        
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.embedding_dim)
        tgt_embedded = self.pos_encoder(tgt_embedded)
        tgt_embedded = tgt_embedded.transpose(0, 1)
        
        if teacher_output is not None and random.random() > 1 / teachers_input_student_ratio:
            # use teacher's encoder output as memory
            memory = teacher_output.detach()
        else:
            memory = self.transformer.encoder(
                src_embedded,
                src_key_padding_mask=src_pad_mask
            )
        
        tgt_mask = self.generate_square_subsequent_mask(tgt_embedded.size(0)).to(src.device)
        
        output = self.transformer.decoder(
            tgt_embedded,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        output = self.generator(output)
        return output.transpose(0, 1)  # shape: [batch_size, tgt_seq_len, vocab_size]

# teacher-student network architecture
class TeacherStudentNetwork(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        nhead,
        hidden_dim,
        teacher_num_layers,
        student_num_layers,
        teacher_weights_path,
        dropout,
    ):
        super(TeacherStudentNetwork, self).__init__()
        self.teacher = Teacher(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_layers=teacher_num_layers,
            dropout=dropout
        )

        self.student = Student(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_layers=student_num_layers,
            dropout=dropout
        )
        
        self.teacher.load_state_dict(torch.load(teacher_weights_path))
        self.teacher.eval()  # teacher only runs inference

    def forward(self, src, tgt, teachers_input_student_ratio=10, src_pad_mask=None, tgt_pad_mask=None):
        with torch.no_grad():
            teacher_output = self.teacher.transformer.encoder(
                self.teacher.pos_encoder(self.teacher.embedding(src) * math.sqrt(self.teacher.embedding_dim)).transpose(0,1),
                src_key_padding_mask=src_pad_mask
            )
        
        student_output = self.student(
            src, tgt, teacher_output, src_pad_mask, tgt_pad_mask, teachers_input_student_ratio
        )
        return student_output

# data preparation
def collate_fn(batch, tokenizer, max_length):
    src_texts = [item['question'] for item in batch]
    tgt_texts = [item['answer'] for item in batch]
    
    src_encodings = tokenizer(src_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    tgt_encodings = tokenizer(tgt_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

    input_ids = src_encodings['input_ids']
    attention_mask = src_encodings['attention_mask']
    labels = tgt_encodings['input_ids']
    labels_attention_mask = tgt_encodings['attention_mask']
    
    return input_ids, labels, attention_mask, labels_attention_mask

# training function
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt, src_mask, tgt_mask in dataloader:
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

# evaluation function
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

# main execution block
if __name__ == "__main__":
    # hyperparams (keeping same for teacher and student models for now)
    embedding_dim = 256
    nhead = 8
    hidden_dim = 512
    teacher_num_layers = 6
    student_num_layers = 3
    dropout = 0.1
    max_seq_length = 128
    batch_size = 16
    num_epochs = 5
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = load_dataset('gsm8k', 'main')
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

    # init teacher model
    teacher = Teacher(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        nhead=nhead,
        hidden_dim=hidden_dim,
        num_layers=teacher_num_layers,
        dropout=dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=learning_rate)

    # train teacher model
    print("now starting training for teacher model...")
    for epoch in range(num_epochs):
        train_loss = train_model(teacher, train_loader, optimizer, criterion, device)
        print(f"epoch {epoch+1}/{num_epochs}, loss: {train_loss:.4f}")

    # save teacher model
    teacher_weights_path = "teacher_model.pth"
    torch.save(teacher.state_dict(), teacher_weights_path)

    # init student model
    student = Student(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        nhead=nhead,
        hidden_dim=hidden_dim,
        num_layers=student_num_layers,
        dropout=dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

    # train student model with KD
    print("now starting training for student model with KD...")
    for epoch in range(num_epochs):
        student.train()
        total_loss = 0
        for src, tgt, src_mask, tgt_mask in train_loader:
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

        optimizer.zero_grad()
        output = student(
            src, tgt_input, teacher_output=teacher_memory,
            src_pad_mask=(src_mask == 0), tgt_pad_mask=(tgt_mask == 0)
        )
        
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"epoch {epoch+1}/{num_epochs}, loss: {avg_loss:.4f}")

    # evaluate models
    print("now starting evaluation for teacher and student models...")
    teacher_loss = evaluate_model(teacher, test_loader, criterion, device)
    student_loss = evaluate_model(student, test_loader, criterion, device)
    
    print(f"teacher model test loss: {teacher_loss:.4f}")
    print(f"student model test loss: {student_loss:.4f}")

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
    
    optimizer = torch.optim.Adam(t_s_network.student.parameters(), lr=learning_rate)

    # train teacher-student network
    print("now starting training for teacher-student network...")
    for epoch in range(num_epochs):
        t_s_network.student.train()
        total_loss = 0
        for src, tgt, src_mask, tgt_mask in train_loader:
            src = src.to(device)
            tgt_input = tgt[:, :-1].to(device)
            tgt_output = tgt[:, 1:].to(device)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask[:, :-1].to(device)

            optimizer.zero_grad()
            output = t_s_network(
                src, tgt_input,
                src_pad_mask=(src_mask == 0), tgt_pad_mask=(tgt_mask == 0)
            )
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"epoch {epoch+1}/{num_epochs}, loss: {avg_loss:.4f}")

    # evaluate teacher-student network
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
    print("\summary:")
    print(f"teacher model only test loss: {teacher_loss:.4f}")
    print(f"student model only test loss: {student_loss:.4f}")
    print(f"teacher-student network test loss: {ts_loss:.4f}")
