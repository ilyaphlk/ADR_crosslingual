from torch.utils.tensorboard import SummaryWriter

def train(model, dataloader, cur_epoch, teacher_model=None, sampler=None, logging_interval=10, tensorboard_writer=None):
    '''
    one epoch of training
    '''

    t0 = time.time()

    total_train_loss = 0

    model.train()

    original_lens_batches = []
    labels_batches = []
    preds_batches = []

    for step, batch in enumerate(dataloader, 1):

        if step % logging_interval == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))

        for key, t in batch.items():
            batch[key] = t.to(device)

        if teacher_model is not None:
            batch['teacher_logits'] = teacher_model(**batch).logits.to(device)

        original_lens_batches.append(batch.pop('original_lens', None))
        
        model.zero_grad()        
        result = model(**batch)

        loss = result.loss
        total_train_loss += loss.item()
        loss.backward()

        preds_batches.append(result.logits.max(-1).indices)
        if 'labels' in batch:
            labels_batches.append(batch['labels'])
        else:
            labels_batches.append(batch['teacher_logits'].max(-1).indices)

        optimizer.step()

    avg_train_loss = total_train_loss / len(dataloader)            
    training_time = format_time(time.time() - t0)

    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar('avg loss (train)', avg_train_loss, cur_epoch)
        #tensorboard_writer.add_scalar('time per epoch (train)', training_time, cur_epoch)

        metrics = compute_metrics(labels_batches, preds_batches, original_lens_batches, dataloader.dataset.int2label)
        for metric_name, metric_value in metrics.items():
            tensorboard_writer.add_scalar(metric_name+" (train)", metric_value, cur_epoch)

    print("")
    print("  Average training loss: {0:.4f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))

    return avg_train_loss, training_time


def eval(model, dataloader, cur_epoch, logging_interval=10, tensorboard_writer=None):

    t0 = time.time()
    model.eval()

    total_eval_loss = 0

    original_lens_batches = []
    labels_batches = []
    preds_batches = []

    for batch in dataloader:
        for key, t in batch.items():
            batch[key] = t.to(device) 
        original_lens_batches.append(batch.pop('original_lens', None))

        with torch.no_grad():
            result = model(**batch)

        loss = result.loss
        total_eval_loss += loss.item()

        preds_batches.append(result.logits.max(-1).indices)
        labels_batches.append(batch['labels'])

    avg_val_loss = total_eval_loss / len(dataloader)
    validation_time = format_time(time.time() - t0)


    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar('avg loss (test)', avg_val_loss, cur_epoch)
        #tensorboard_writer.add_scalar('time per epoch (test)', validation_time, cur_epoch)
        
        metrics = compute_metrics(labels_batches, preds_batches, original_lens_batches, dataloader.dataset.int2label)
        for metric_name, metric_value in metrics.items():
            tensorboard_writer.add_scalar(metric_name+" (test)", metric_value, cur_epoch)
        
    
    print("  Test Loss: {0:.4f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    return avg_val_loss, validation_time

