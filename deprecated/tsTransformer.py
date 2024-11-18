import random
import time
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a long enough P matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerForward(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(TransformerForward, self).__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=1  # One layer per TransformerForward module
        )

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        return self.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)


class Teacher(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, hidden_dim, num_layers, num_classes, dropout, max_seq_length=512):
        super(Teacher, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=max_seq_length)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, 
                                                    dim_feedforward=hidden_dim, dropout=dropout, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 30),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(30, num_classes)
        )

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        embedded = self.embedding(x)  # Shape: [batch_size, seq_len, embedding_dim]
        embedded = embedded.transpose(0, 1)  # Shape: [seq_len, batch_size, embedding_dim]
        embedded = self.pos_encoder(embedded)
        
        cls_tokens = self.cls_token.expand(-1, embedded.size(1), -1)  # Shape: [1, batch_size, embedding_dim]
        transformer_input = torch.cat((cls_tokens, embedded), dim=0)  # Shape: [seq_len + 1, batch_size, embedding_dim]
        
        transformer_out = self.transformer(transformer_input, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        cls_output = transformer_out[0]  # [batch_size, embedding_dim]
        
        out_1 = transformer_out  # All transformer outputs
        out_2 = transformer_out  # Could include intermediate representations if needed
        out_3 = cls_output  # CLS token
        out_4 = self.classifier(cls_output)
        
        return out_1, out_2, out_3, out_4


class Student(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, hidden_dim, num_layers, num_classes, dropout, max_seq_length=512):
        super(Student, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=max_seq_length)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, 
                                                    dim_feedforward=hidden_dim, dropout=dropout, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 30),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(30, num_classes)
        )

    def forward(self, x, teacher_out_1=None, teacher_out_2=None, teachers_input_student_ratio=10, src_mask=None, src_key_padding_mask=None):
        embedded = self.embedding(x)  # Shape: [batch_size, seq_len, embedding_dim]
        embedded = embedded.transpose(0, 1)  # Shape: [seq_len, batch_size, embedding_dim]
        embedded = self.pos_encoder(embedded)
        
        cls_tokens = self.cls_token.expand(-1, embedded.size(1), -1)  # Shape: [1, batch_size, embedding_dim]
        transformer_input = torch.cat((cls_tokens, embedded), dim=0)  # Shape: [seq_len + 1, batch_size, embedding_dim]
        
        out_1 = transformer_input
        out_2 = transformer_input  # Can adjust based on teacher's outputs
        
        # Incorporate teacher's outputs if available
        if teacher_out_1 is not None and random.random() > 1 / teachers_input_student_ratio:
            out_2 = teacher_out_1.detach()
        else:
            out_2 = out_1

        transformer_out = self.transformer(out_2, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        cls_output = transformer_out[0]  # [batch_size, embedding_dim]
        
        if teacher_out_2 is not None and random.random() > 1 / teachers_input_student_ratio:
            out_3 = teacher_out_2.detach()
        else:
            out_3 = cls_output
        
        out_4 = self.classifier(out_3)
        
        return out_1, out_2, out_3, out_4


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
        num_classes,
        dropout,
        max_seq_length=512,
    ):
        super(TeacherStudentNetwork, self).__init__()
        self.teacher = Teacher(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_layers=teacher_num_layers,
            num_classes=num_classes,
            dropout=dropout,
            max_seq_length=max_seq_length
        )

        self.student = Student(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_layers=student_num_layers,
            num_classes=num_classes,
            dropout=dropout,
            max_seq_length=max_seq_length
        )
        
        self.teacher.load_state_dict(torch.load(teacher_weights_path))

    def forward(self, x, teachers_input_student_ratio=10, src_mask=None, src_key_padding_mask=None):
        with torch.no_grad():
            t_out_1, t_out_2, t_out_3, t_out_4 = self.teacher(x, src_mask, src_key_padding_mask)
        s_out_1, s_out_2, s_out_3, s_out_4 = self.student(
            x, t_out_1, t_out_2, teachers_input_student_ratio, src_mask, src_key_padding_mask
        )
        return (
            [t_out_1.detach(), t_out_2.detach(), t_out_3.detach(), t_out_4.detach()],
            [s_out_1, s_out_2, s_out_3, s_out_4]
        )


if __name__ == "__main__":
    vocab_size = 10000  # Example vocabulary size
    embedding_dim = 128
    nhead = 8
    hidden_dim = 512
    teacher_num_layers = 6
    student_num_layers = 3
    num_classes = vocab_size  # Typically, number of classes equals vocab size for generation
    dropout = 0.2
    max_seq_length = 32

    # Example input: batch_size=2, sequence_length=32
    input_seq = torch.randint(0, vocab_size, (2, max_seq_length))  # Shape: [batch_size, seq_len]
    
    # Initialize teacher and student
    t = Teacher(vocab_size, embedding_dim, nhead, hidden_dim, teacher_num_layers, num_classes, dropout, max_seq_length)
    s = Student(vocab_size, embedding_dim, nhead, hidden_dim, student_num_layers, num_classes, dropout, max_seq_length)
    
    # Forward pass through teacher
    start = time.time()
    t_out_1, t_out_2, t_out_3, t_out_4 = t(input_seq)
    print(f"time taken for teacher = {time.time() - start} seconds")
    
    # Forward pass through student
    start = time.time()
    s_out = s(input_seq, t_out_1, t_out_2)
    print(f"time taken for student = {time.time() - start} seconds")
    
    # Save teacher model
    path = "model/teacher_model.bin"
    torch.save(t.state_dict(), path)
    
    # Initialize Teacher-Student Network
    t_s = TeacherStudentNetwork(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        nhead=nhead,
        hidden_dim=hidden_dim,
        teacher_num_layers=teacher_num_layers,
        student_num_layers=student_num_layers,
        teacher_weights_path=path,
        num_classes=num_classes,
        dropout=dropout,
        max_seq_length=max_seq_length
    )
    
    # Forward pass through Teacher-Student Network
    start = time.time()
    x = t_s(input_seq)
    print(f"time taken for teacher-student = {time.time() - start} seconds")