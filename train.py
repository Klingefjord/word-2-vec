
import torch
from torch import nn
from data.wikitext import data_loader, setup_data
from tqdm.auto import tqdm
from utils.save import save_model
from utils.plot import plot_losses
from utils.eval import eval

def train(model, epochs, criterion, collate_fn, device):
    """Main training loop"""
    losses = []
    model = model.to(device)
    criterion = criterion.to(device)

    for e in range(epochs):
        epoch_losses = []
        batch_generator = collate_fn()

        for i, batch in enumerate(tqdm(batch_generator)):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            input = batch[:,0,:]
            target = batch[:,1,:]
            output = model(input)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            with torch.no_grad:
                epoch_losses.append(loss.detach().clone().item())

        print(f"epoch {e}: loss {sum(epoch_losses) / len(epoch_losses)}")
        save_model(model, batch_size, vocab_size, embedding_size, n_docs, epoch=e)

    # return a list of flattened losses
    return [loss for epoch_losses in t for loss in epoch_losses]

if __name__ == "__main__":
    """Train the model using the data and hyper parameters specified in the config.json file"""
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    configuration = parse_configuration(config_file)
    epochs = configuration['hyper_params']['epochs']
    learning_rate = configuration['hyper_params']['learning_rate']
    batch_size = configuration['train_dataset_params']['loader_params']['batch_size']
    vocab_size = configuration['train_dataset_params']['loader_params']['vocab_size']

    print("Setting up vocabulary... 📕")
    df, vocab, vocab_strings = setup_data()
    
    print("Setting up model... ⏱")
    model = Transformer(vocab_size, vocab_size, d_model, d_hidden, n_heads, N)
    optimizer = NoamOptimizer(torch.optim.Adam(model.parameters(), betas=(.9,.98), eps=1e-9, lr=0.), d_model, warmup_steps)
    criterion = LabelSmoothingCrossEntropy().to(device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _collate_fn():
        return data_loader(df, batch_size)

    print(f"✨ Training model with \
        epochs={epochs} \
        batch_size={batch_size} \
        vocab_size={vocab_size} \
        warmup_steps={warmup_steps} \
        training_examples={len(train_loader) * batch_size} \
        on device={device}")        
    train_losses, valid_losses = train(model, epochs, criterion, _collate_fn, device)

    print("🏅 Model finished training, plotting losses...")
    plot_losses(train_losses, valid_losses)

    eval(model, vocab, vocab_strings, vocab_size)

