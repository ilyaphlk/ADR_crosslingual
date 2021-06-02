import time
import torch
from torch.utils.tensorboard import SummaryWriter
from ADR_crosslingual.utils import format_time#, compute_metrics


def train_model(model, dataloader, cur_epoch, device, optimizer,
          teacher_model=None, sampler=None,
          logging_interval=10, tensorboard_writer=None, tb_postfix=" (train)", print_progress=True,
          compute_metrics=None, int2label=None):
    '''
    one epoch of training
    model - model to train
    dataloader - dataloader object, from which to get batches
    cur_epoch - needed for logging
    teacher_model - if not None, then use output logits of this model as targets
    sampler - if not None, use batch sampling to get best samples from batch via teacher model
    '''

    t0 = time.time()

    total_train_loss = 0

    model.train()

    #original_lens_batches = []
    #labels_batches = []
    #preds_batches = []
    label_ids = []
    preds_ids = []

    #label_ids = unpack(labels, original_lens) # list of (list of true labels for doc)
    #preds_ids = unpack(preds, original_lens)

    for step, batch in enumerate(dataloader, 1):

        if print_progress and step % logging_interval == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))

        del step

        original_lens_batch = batch.pop('original_lens', None)
        device = next(model.parameters()).device

        # ?TODO explicitly move batch to cpu?

        if sampler is not None:
            batch = sampler(batch, teacher_model)  # possible reduction of the whole batch

        if teacher_model is not None and 'teacher_logits' not in batch:
            teacher_model.eval()
            teacher_device = next(teacher_model.parameters()).device
            for key, t in batch.items():
                batch[key] = t.to(teacher_device)
            batch['teacher_logits'] = teacher_model(**batch).logits

        # move batch yet again
        for key, t in batch.items():
            batch[key] = t.to(device)
        
        model.zero_grad()        
        result = model(**batch)

        loss = result.loss
        total_train_loss += float(loss)

        '''
        if L2_C > 0:
            for layer, layer_init in zip(model.bert_like.encoder.named_parameters(),
                                        model_initial.bert_like.encoder.named_parameters()):
                encoder_layer_name = layer[0]
                encoder_layer_number = int(encoder_layer_name.split(".")[1])
                L2_delta = L2_C * ((layer[1] - layer_init[1])**2).sum() / L2_exp ** (encoder_layer_number + 1)
                loss += L2_delta
        '''

        loss.backward()
        del loss

        if original_lens_batch is not None:
            if 'labels' in batch:
                label_ids.extend(unpack([batch['labels']], [original_lens_batch]))
            else:
                label_ids.extend(unpack([batch['teacher_logits'].max(-1).indices], [original_lens_batch]))
            preds_ids.extend(unpack([result.logits.max(-1).indices],  [original_lens_batch]))

        del original_lens_batch

        optimizer.step()

    avg_train_loss = total_train_loss / len(dataloader)            
    training_time = format_time(time.time() - t0)

    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar('avg loss'+tb_postfix, avg_train_loss, cur_epoch)
        if compute_metrics is not None:
            int2label = dataloader.dataset.int2label if int2label is None else int2label
            metrics = compute_metrics(label_ids, preds_ids, int2label)
            tensorboard_writer.add_scalars("metrics"+tb_postfix, metrics, cur_epoch)

    if print_progress:
        print("")
        print("  Average training loss: {0:.4f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

    return avg_train_loss, training_time


def eval_model(model, dataloader, cur_epoch, device,
         logging_interval=10, tensorboard_writer=None, tb_postfix=" (test)", print_progress=True,
         compute_metrics=None, int2label=None):

    t0 = time.time()
    model.eval()

    total_eval_loss = 0

    #original_lens_batches = []
    #labels_batches = []
    #preds_batches = []

    label_ids = []
    preds_ids = []

    for batch in dataloader:
        for key, t in batch.items():
            batch[key] = t.to(device) 
        original_lens_batch = batch.pop('original_lens', None)

        with torch.no_grad():
            result = model(**batch)

            total_eval_loss += float(result.loss)

            label_ids.extend(unpack([batch['labels']], [original_lens_batch]))
            preds_ids.extend(unpack([result.logits.max(-1).indices], [original_lens_batch]))
            del original_lens_batch

            del result

    avg_val_loss = total_eval_loss / len(dataloader)
    validation_time = format_time(time.time() - t0)


    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar('avg loss'+tb_postfix, avg_val_loss, cur_epoch)
        if compute_metrics is not None:
            int2label = dataloader.dataset.int2label if int2label is None else int2label
            metrics = compute_metrics(label_ids, preds_ids, int2label)
            tensorboard_writer.add_scalars("metrics"+tb_postfix, metrics, cur_epoch)
        
    if print_progress:
        print("  Test Loss: {0:.4f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

    return avg_val_loss, validation_time
