import numpy as np
import torch
from ADR_crosslingual.utils import collate_dicts
from torch.nn.utils.rnn import pad_sequence


class BaseUncertaintySampler:
    def __init__(
        self,
        strategy='confident',
        n_samples_out=1,
        scoring_batch_sz=1,
        averaging_share=None,
        return_vars=False):
        '''
          strategy - whether to return samples in which the model is confident the most, the least or inbetween
          n_samples_out - how many samples should be selected from the batch
        '''
        assert strategy in ['confident', 'mid', 'uncertain']
        self.strategy = strategy
        self.n_samples_out = n_samples_out
        self.stochastic = None
        self.n_forward_passes = 1
        self.scoring_batch_sz = scoring_batch_sz
        if type(averaging_share) != float:
            self.averaging_share = None
        else:
            self.averaging_share = averaging_share
        self.return_vars = return_vars


    def __call__(self, batch, model):
        scoring_batch_sz = self.scoring_batch_sz

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

        original_lens = batch.pop('original_lens', None)

        for i in range(0, N, scoring_batch_sz):
            samples = []
            for j in range(i, min(N, i+scoring_batch_sz)):
                samples.append({key : val[j,:] for key, val in batch.items()})

            batched_samples = collate_dicts(samples)
            del samples

            for key, t in batched_samples.items():
                batched_samples[key] = t.to(device)

            probs_list = []
            for _ in range(self.n_forward_passes):
                with torch.no_grad():
                    probs = torch.nn.functional.softmax(
                        model(**batched_samples).logits.to('cpu'),
                        dim=-1
                    )
                    probs_list.append(probs)

            del batched_samples

            probs = torch.stack(probs_list, dim=1)  # shape = (B, T, N, C)
            del probs_list
            for j in range(probs.size(0)):
                #truncate probs:
                cur_probs = probs[j,:,:,:]
                if original_lens is not None:
                    orig_len = original_lens[i+j]
                    cur_probs = cur_probs[:, :orig_len, :]

                token_scores = self._calculate_uncertainty_score(cur_probs)
                aggregated_score = None
                if self.averaging_share is None:
                    aggregated_score = torch.mean(token_scores)
                else:
                    k = int(token_scores.size(0) * self.averaging_share)
                    k = max(1, k)
                    aggregated_score = torch.topk(token_scores, k, dim=0).values.mean()

                scores.append(aggregated_score)
                del cur_probs
            
            del probs

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            '''
            if not self.stochastic:
                computed_logits.append({'teacher_logits':logits})
            '''

        scores = np.array(scores)
        idx_sorted = np.argsort(scores)

        if self.strategy == 'confident':
            idx_selected = idx_sorted[:self.n_samples_out]
        elif self.strategy == 'uncertain':
            idx_selected = idx_sorted[-self.n_samples_out:]
        elif self.strategy == 'mid':
            raise NotImplementedError

        #filtered_batch = {key : val[idx_selected,:].to(device) for key, val in batch.items()}

        ############
        # repack batch - slow
        samples = []
        for idx in idx_selected:
            samples.append({key : val[idx,:] for key, val in batch.items()})

        filtered_batch = collate_dicts(samples, return_lens=False)
        del samples
        for key, t in filtered_batch.items():
            t.to(device)

        ############

        #######
        # TODO: instead, find max len in new batch, truncate filtered
        #######


        original_lens = original_lens[idx_selected]
        filtered_batch['original_lens'] = original_lens.to(device)


        for key, t in filtered_batch.items():
            print(key, t.size())


        ###############
        ### explicitly compute prediction variances
        ###############

        if self.return_vars:

            samples_variances = []
            model.train()

            probs_list = []
            for _ in range(self.n_forward_passes):
                with torch.no_grad():
                    probs = torch.nn.functional.softmax(
                        model(**filtered_batch).logits.to('cpu'),
                        dim=-1
                    )
                    probs_list.append(probs)

            probs = torch.stack(probs_list, dim=1)  # shape = (B, T, N, C)
            del probs_list

            print("probs size:", probs.size())

            var_sampler = VarianceSampler('uncertain', 1)  # should make a static method instead
            for j in range(probs.size(0)):
                #truncate probs:
                cur_probs = probs[j,:,:,:]
                orig_len = original_lens[j] # TODO sketchy?
                cur_probs = cur_probs[:, :orig_len, :]

                print("cur probs shape:", cur_probs.size())

                token_variances = var_sampler._calculate_variances(cur_probs)
                
                samples_variances.append(torch.tensor(token_variances))
                del cur_probs
            
            del probs

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # TODO collate variances

            filtered_batch['samples_variances'] = pad_sequence(samples_variances,
                                                                batch_first=True,
                                                                padding_value=0).to(device)  # samples_variances to tensor

            print("variances batch shape:", filtered_batch['samples_variances'].size())

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
        margins = margins.float()
        
        return -margins


class LeastConfidenceSampling(BaseUncertaintySampler):
    def _calculate_uncertainty_score(self, probs):
        probs = probs.squeeze()
        highest_probs = torch.topk(probs, 1, dim=-1).values

        return (1 - highest_probs).float()


class RatioOfConfidence(BaseUncertaintySampler):
    def _calculate_uncertainty_score(self, probs, eps=1e-3):
        probs = probs.squeeze()
        highest_probs = torch.topk(probs, 2, dim=-1).values
        ratios = highest_probs[:, 0] / (highest_probs[:, 1] + eps)

        return -ratios.float()


class EntropySampler(BaseUncertaintySampler):
    def _calculate_uncertainty_score(self, probs, eps=1e-3):
        probs = probs.squeeze()
        entropies = -probs * torch.log2(probs)
        return entropies.float()


class RandomSampler(BaseUncertaintySampler):
    def _calculate_uncertainty_score(self, probs):
        return torch.tensor(np.random.uniform(size=probs.size(0))).float()


class BALDSampler(BaseUncertaintySampler):
    def __init__(
        self,
        strategy,
        n_samples_out,
        n_forward_passes=5,
        scoring_batch_sz=1,
        averaging_share=None,
        return_vars=False):
        super().__init__(strategy, n_samples_out, scoring_batch_sz, averaging_share, return_vars)
        self.n_forward_passes = n_forward_passes
        self.stochastic = True
        self.scoring_batch_sz = scoring_batch_sz

    def _calculate_uncertainty_score(self, probs):
        mean_entropy_of_pred = -((probs * torch.log(probs)).sum(dim=-1)).mean(dim=0)
        entropy_of_mean_pred = -((probs.mean(dim=0))*torch.log(probs.mean(dim=0))).sum(dim=-1)
        token_scores = entropy_of_mean_pred - mean_entropy_of_pred
        return token_scores


class VarianceSampler(BaseUncertaintySampler):
    def __init__(
        self,
        strategy,
        n_samples_out,
        n_forward_passes=5,
        scoring_batch_sz=1,
        averaging_share=None,
        return_vars=False):
        super().__init__(strategy, n_samples_out, scoring_batch_sz, averaging_share, return_vars)
        self.n_forward_passes = n_forward_passes
        self.stochastic = True

    def _calculate_uncertainty_score(self, probs):
        Vars = self._calculate_variances(probs)
        return Vars

    def _calculate_variances(self, probs):
        # probs.shape = (T, N, C)
        E = probs.mean(dim=0)  # shape = (N, C)
        Means = torch.diag(E @ E.T)  # shape = (N,)
        P = probs @ probs.transpose(-1, -2)  # shape = (T, N, N)
        P = P[:, np.arange(P.size(1)), np.arange(P.size(1))]  # shape = (T, N) - taking diagonals in a batch
        Vars = (P - Means).mean(dim=0)
        return Vars