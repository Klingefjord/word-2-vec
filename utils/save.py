import os
import torch

def save_model(model, batch_size, vocab_size, embedding_size, n_docs, epoch=None):
    if not os.path.exists('./models/'):
        os.mkdir('./models/')

    torch.save(model.state_dict(), f'./models/bs-{batch_size} \
        _vs-{vocab_size} \
        _es-{embedding_size} \
        _nd-{n_docs} \
        _e-{"None" if epoch is None else epoch}")