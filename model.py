"""
TODOs:
  - need to add change teacher-student network to use MDP
"""

import torch
import math
import random
import torch.nn as nn

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
    def __init__(self, vocab_size, embedding_dim=128, nhead=4, hidden_dim=256, num_layers=2, dropout=0.05, teacher_embedding_dim=256):
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
        
        # Linear layer to project teacher's output to student's embedding dimension
        self.teacher_to_student_proj = nn.Linear(teacher_embedding_dim, embedding_dim)

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
            memory = self.teacher_to_student_proj(teacher_output.detach())
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