import numpy as np
import torch


class BaseUncertaintySampler:
    def __init__(self, strategy, n_samples_out):
        '''
          strategy - whether to return samples in which the model is confident the most, the least or inbetween
          n_samples_out - how many samples should be selected from the batch
        '''
        assert strategy in ['most', 'mid', 'least']
        self.strategy = strategy
        self.n_samples_out = n_samples_out


    def __call__(self, batch, model):
        if batch.shape[0] <= self.n_samples_out:  # no need to select samples
            return batch

        scores = []
        for i in batch.shape[0]:
            sample = {key : val[i,:].to(device) for key, val in batch.items()}
            logits = model(**sample).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            scores.append(self._calculate_uncertainty_score(probs))

        scores = np.array(scores)
        idx_sorted = np.argsort(scores)
        if self.strategy is 'most':
            idx_selected = idx_sorted[:self.n_samples_out]
        elif self.startegy is 'least':
            idx_selected = idx_sorted[-self.n_samples_out:]
        elif self.strategy is 'mid':
            raise NotImplementedError

        return {key : val[idx_selected,:] for key, val in batch.items()}


    def _calculate_uncertainty_score(self, probs):
        '''
          the higher the score, the less confident the model is
          probs - tensor of shape (n_tokens, n_samples)
        '''
        pass


class MarginOfConfidenceSampler(BaseUncertaintySampler):
    def _calculate_uncertainty_score(self, probs):
        probs = probs.squeeze()
        highest_probs = torch.topk(probs, 2, dim=-1).values
        margins = highest_probs[:, 0] - highest_probs[:, 1]
        mean_margin = margins.float().mean()
        
        return mean_margin

