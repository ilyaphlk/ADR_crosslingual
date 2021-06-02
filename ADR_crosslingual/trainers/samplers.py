import numpy as np
import torch
from ADR_crosslingual.utils import collate_dicts


class BaseUncertaintySampler:
    def __init__(self, strategy='confident', n_samples_out=1):
        '''
          strategy - whether to return samples in which the model is confident the most, the least or inbetween
          n_samples_out - how many samples should be selected from the batch
        '''
        assert strategy in ['confident', 'mid', 'uncertain']
        self.strategy = strategy
        self.n_samples_out = n_samples_out
        self.stochastic = None
        self.n_forward_passes = 1


    def __call__(self, batch, model):

        model.eval()
        device = next(model.parameters()).device

        N = batch['input_ids'].shape[0]
        if N <= self.n_samples_out:  # no need to select samples
            for key, t in batch.items():
                batch[key] = t.to(device)
            batch['teacher_logits'] = model(**batch).logits.to(device)
            return batch

        if self.stochastic:  # in order to use MC Dropout
            model.train()

        scores = []
        computed_logits = []

        for i in range(N):
            sample = {key : val[i,:].to(device).unsqueeze(0) for key, val in batch.items()}
            probs_list = []
            logits = None
            for _ in range(self.n_forward_passes):
                #logits = model(**sample).logits.squeeze()
                with torch.no_grad():
                    probs = torch.nn.functional.softmax(
                        model(**sample).logits.squeeze().to('cpu'),
                        dim=-1
                    )
                    probs_list.append(probs)

            probs = torch.stack(probs_list, dim=0)
            scores.append(self._calculate_uncertainty_score(probs))

            del sample
            torch.cuda.empty_cache()
            '''
            if not self.stochastic:
                computed_logits.append({'teacher_logits':logits})
            '''

        scores = np.array(scores)
        idx_sorted = np.argsort(scores)

        print(self.strategy)
        if self.strategy == 'confident':
            print("found")
            idx_selected = idx_sorted[:self.n_samples_out]
        elif self.strategy == 'uncertain':
            idx_selected = idx_sorted[-self.n_samples_out:]
        elif self.strategy == 'mid':
            raise NotImplementedError

        filtered_batch = {key : val[idx_selected,:] for key, val in batch.items()}

        '''
        if not self.stochastic:
            computed_logits_batch = collate_dicts(computed_logits, return_lens=False)['teacher_logits']
            filtered_batch['teacher_logits'] = computed_logits_batch[idx_selected,:]
        '''

        return filtered_batch


    def get_student_batch(self, batch, model, teacher_batch_sz=1):
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
        
        return -mean_margin


class LeastConfidenceSampling(BaseUncertaintySampler):
    def _calculate_uncertainty_score(self, probs):
        probs = probs.squeeze()
        highest_probs = torch.topk(probs, 1, dim=-1).values
        mean_difference = (1 - highest_probs).float().mean()

        return mean_difference


class RatioOfConfidence(BaseUncertaintySampler):
    def _calculate_uncertainty_score(self, probs, eps=1e-3):
        probs = probs.squeeze()
        highest_probs = torch.topk(probs, 2, dim=-1).values
        ratios = highest_probs[:, 0] / (highest_probs[:, 1] + eps)
        mean_ratio = ratios.float().mean()

        return -mean_ratio


class EntropySampler(BaseUncertaintySampler):
    def _calculate_uncertainty_score(self, probs, eps=1e-3):
        probs = probs.squeeze()
        entropies = -probs * torch.log2(probs)
        return entropies.float().mean()


class RandomSampler(BaseUncertaintySampler):
    def _calculate_uncertainty_score(self, probs):
        return np.random.normal()


class BALDSampler(BaseUncertaintySampler):
    def __init__(self, strategy, n_samples_out, n_forward_passes):
        super().__init__(strategy, n_samples_out)
        self.n_forward_passes = n_forward_passes
        self.stochastic = True

    def _calculate_uncertainty_score(self, probs):
        mean_entropy_of_pred = -((probs * torch.log(probs)).sum(dim=-1)).mean(dim=0)
        entropy_of_mean_pred = -((probs.mean(dim=0))*torch.log(probs.mean(dim=0))).sum(dim=-1)
        token_scores = entropy_of_mean_pred - mean_entropy_of_pred
        return token_scores.mean()


class VarianceSampler(BaseUncertaintySampler):
    def __init__(self, strategy, n_samples_out, n_forward_passes):
        super().__init__(strategy, n_samples_out)
        self.n_forward_passes = n_forward_passes
        self.stochastic = True

    def _calculate_uncertainty_score(self, probs):
        Vars = self._calculate_variances(probs)
        return Vars.mean()

    def _calculate_variances(self, probs):
        # probs.shape = (T, N, C)
        E = probs.mean(dim=0)  # shape = (N, C)
        Means = torch.diag(E @ E.T)  # shape = (N,)
        P = probs @ probs.transpose(1, 2)  # shape = (T, N, N)
        P = P[:, np.arange(P.size(1)), np.arange(P.size(1))]  # shape = (T, N) - taking diagonals in a batch
        Vars = (P - Means).mean(dim=0)
        return Vars