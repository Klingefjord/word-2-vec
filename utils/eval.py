from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def _tsne_plot(model, vocab, vocab_strings):
    """
    Use t-SNE to reduce the 300 dimensions of the embedding vectors 
    for 200 random words to 2D and plot the results
    """
    labels = []
    tokens = []
    
    for _ in range(200):
        word = vocab_strings[random.randint(0, 2000)]
        tokens.append(model.embedding_matrix(torch.zeros(vocab_size).scatter_(0, torch.tensor(vocab[word]), 1.)).tolist())
        labels.append(word)
        
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    values = tsne_model.fit_transform(tokens)
    
    x = []
    y = []
    for value in values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16,16))
    
    for i in range(len(values)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        
    plt.show()

def eval(model, vocab, vocab_strings, vocab_size):
    # Cosine distances
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    queen = torch.zeros(vocab_size).scatter_(0, torch.tensor(vocab['queen']), 1.)
    king = torch.zeros(vocab_size).scatter_(0, torch.tensor(vocab['king']), 1.)
    banana_for_scale = torch.zeros(vocab_size).scatter_(0, torch.tensor(vocab['banana']), 1.)

    print(f"Cosine distance between queen and king: {cos(model.embedding_matrix(queen), model.embedding_matrix(king))}")
    print(f"Cosine distance between queen and banana (for scale): {cos(model.embedding_matrix(queen), model.embedding_matrix(banana_for_scale))}")

    # t-SNE plotting
    print("Plotting words using t-SNE...")
    _tsne_plot(model, vocab, vocab_strings)