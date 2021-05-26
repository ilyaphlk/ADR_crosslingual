import numpy as np
import torch
from ADR_crosslingual.utils import collate_dicts


class BaseUncertaintySampler:
    def __init__(self, strategy='most', n_samples_out=1):
        '''
          strategy - whether to return samples in which the model is confident the most, the least or inbetween
          n_samples_out - how many samples should be selected from the batch
        '''
        assert strategy in ['most', 'mid', 'least']
        self.strategy = strategy
        self.n_samples_out = n_samples_out


    def __call__(self, batch, model):
        device = next(model.parameters()).device

        N = batch['input_ids'].shape[0]
        if N <= self.n_samples_out:  # no need to select samples
            for key, t in batch.items():
                batch[key] = t.to(device)
            batch['teacher_logits'] = model(**batch).logits.to(device)
            return batch

        scores = []
        computed_logits = []

        for i in range(N):
            sample = {key : val[i,:].to(device).unsqueeze(0) for key, val in batch.items()}
            logits = model(**sample).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            scores.append(self._calculate_uncertainty_score(probs))
            computed_logits.append({'teacher_logits':logits.squeeze()})

        scores = np.array(scores)
        idx_sorted = np.argsort(scores)
        if self.strategy is 'most':
            idx_selected = idx_sorted[:self.n_samples_out]
        elif self.startegy is 'least':
            idx_selected = idx_sorted[-self.n_samples_out:]
        elif self.strategy is 'mid':
            raise NotImplementedError

        filtered_batch = {key : val[idx_selected,:] for key, val in batch.items()}
        computed_logits_batch = collate_dicts(computed_logits, return_lens=False)['teacher_logits']
        filtered_batch['teacher_logits'] = computed_logits_batch[idx_selected,:]

        return filtered_batch


    def get_student_batch():
        pass


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