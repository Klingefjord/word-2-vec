import torch
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import pandas as pd
from fastai.data.external import untar_data, URLs


def setup_data(vocab_size, min_frequency):
    def _create_vocab(df, vocab_size, min_frequency=1):
        counter = Counter()
        print(f"Starting parsing docs, in total {len(df.values)}")
        for _, doc in tqdm(enumerate(df.values.tolist())):
            doc_counter = Counter([token.text for token in tokenizer(doc) if not token.is_stop and token.is_alpha])
            counter += doc_counter

        vocab_strings = [token for token, count in counter.most_common(vocab_size) if  count >= min_frequency]
        # create a dictionary with a default of -1 for word not existing in our vocab
        vocab = defaultdict(lambda: -1, { value: key for key, value in enumerate(vocab_strings)})
        print(f"Created vocab of size {len(vocab)}. Most common words are {vocab_strings[:10]}")
        return vocab, vocab_strings

    path = untar_data(URLs.WIKITEXT)
    df = pd.read_csv(path/'train.csv', header=None).apply(lambda x: x[0], axis=1)
    vocab, vocab_strings = _create_vocab(df, vocab_size, min_frequency)
    return df, vocab, vocab_strings


def data_loader(df, batch_size, num_workers=1):
    def _tokenizer(df):
        nlp = English()
        tokenizer = Tokenizer(nlp.vocab)

        for doc in tokenizer.pipe(df.values.tolist(), batch_size=50):
            for token in doc:
                yield token

    return DataLoader(WikiDataSet(_tokenizer(df)), batch_size=batch_size, drop_last=True, num_workers=num_workers)


class WikiDataSet(torch.utils.data.IterableDataset):
    """
    PyTorch IterableDataset wrapper for WIKITEXT. 
    Returns a generator yielding a series of skip-gram pairs the size of context_window
    """
    def __init__(self, generator, context_window=5):
        super(WikiDataSet).__init__()
        self.generator = generator
        self.context_window = context_window
        
    def skipgramify(self):
        """
        Get number of valid tokens equal to self.context_window. Take the middle token in the sequence and convert to one-hot vector (this is used as input).
        Take the other tokens and convert to n-hot encoded target vector. Return a stack of the input and target vectors.
        """
        eos = False
        while not eos:
            tokens = []
            while len(tokens) < self.context_window:
                try:
                    text = next(self.generator).text
                    if (vocab[text] != -1):
                        tokens.append(vocab[text])
                except:
                    eos = True
                    break
                    
            if (len(tokens) == self.context_window):
                input_one_hot = torch.zeros(vocab_size).scatter_(0, torch.tensor(tokens[self.context_window//2]), 1.)
                target_one_hot = torch.zeros(vocab_size).scatter_(0, torch.tensor(tokens[:self.context_window//2] + tokens[self.context_window//2+1:]), 1.)
                yield torch.stack([input_one_hot, target_one_hot])
                                                                                                                                                             
    def __iter__(self): 
        return self.skipgramify()