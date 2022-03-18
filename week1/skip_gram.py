# 참고 사이트: https://direction-f.tistory.com/29

class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist=None):
        super().__init__()
        
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist
        
        # define embedding layers for input and output words
        self.in_embed = nn.Embedding(n_vocab,n_embed)
        self.out_embed = nn.Embedding(n_vocab,n_embed)
        
        # Initialize both embedding tables with uniform distribution
        self.in_embed.weight.data.uniform_(-1,1)
        self.out_embed.weight.data.uniform_(-1,1)
        
    def forward_input(self, input_words):
        
        input_vector = self.in_embed(input_words)
        return input_vector
    
    def forward_output(self, output_words):
        output_vector = self.out_embed(output_words)
        return output_vector
    
    def forward_noise(self, batch_size, n_samples):
        """ Generate noise vectors with shape (batch_size, n_samples, n_embed)"""
        if self.noise_dist is None:
            # Sample words uniformly
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist
            
        # Sample words from our noise distribution
        noise_words = torch.multinomial(noise_dist,
                                        batch_size * n_samples,
                                        replacement=True)  ## noise sample 만큼 데이터 생성 
    

        noise_vector = self.out_embed(noise_words).view(batch_size,n_samples,self.n_embed)        
        return noise_vector
        
class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):
        
        batch_size, embed_size = input_vectors.shape
        
        # Input vectors should be a batch of column vectors
        input_vectors = input_vectors.view(batch_size, embed_size, 1)
        
        # Output vectors should be a batch of row vectors
        output_vectors = output_vectors.view(batch_size, 1, embed_size)
        
        # bmm = batch matrix multiplication (b*n*e)(b*e*n) = (b*n*n)
        # correct log-sigmoid loss
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss = out_loss.squeeze()
        
        
        # incorrect log-sigmoid loss
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        ## 각 row별로 loss 합산(e.g. [input-negative1 loss] +[input-negative2 loss]
        noise_loss = noise_loss.squeeze().sum(1)  
        return -(out_loss + noise_loss).mean()


if __name__ == "__main__":
    embedding_dim = 30
    model = SkipGramNeg(len(vocab), embedding_dim, noise_dist =noise_dist)

    criterion = NegativeSamplingLoss()
    optimizer = optim.Adam(model.parameters(), lr =0.03)

    for epoch in range(1000):
        for input_words, target_words in get_batches(data):
            step +=1
            inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)
            input_vectors = model.forward_input(inputs)
            output_vectors = model.forward_output(targets)
            noise_vectors = model.forward_noise(inputs.shape[0], 5)
            
            loss = criterion(input_vectors, output_vectors, noise_vectors)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                print("Epoch: {}/{}".format(epoch+1, 1000))
                print("Loss: ", loss.item())​