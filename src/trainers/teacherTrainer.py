def train_teacher(logging_interval=10):
    '''
    one epoch of training
    '''

    t0 = time.time()

    total_train_loss = 0

    model.train()
    dataloader = train_dataloader
    for step, batch in enumerate(dataloader, 1):

        if step % logging_interval == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))

        model.zero_grad()        
        result = model(**batch)

        loss = result.loss
        total_train_loss += loss.item()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    avg_train_loss = total_train_loss / len(dataloader)            
    
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.4f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))

    return avg_train_loss, training_time
