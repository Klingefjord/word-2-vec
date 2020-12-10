from torch import nn

class Word2VecModel(nn.Module):
    """Simple FFNN with two linear layers"""
    def __init__(self, vocab_size, embedding_size):
        super(Word2VecModel, self).__init__()
        self.embedding_matrix = nn.Linear(vocab_size, embedding_size)
        self.context_matrix = nn.Linear(embedding_size, vocab_size)

    def forward(self, x): 
        return self.context_matrix(self.embedding_matrix(x))

model = Word2VecModel(vocab_size, embedding_size)