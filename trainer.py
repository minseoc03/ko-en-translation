import torch
import math
from tqdm import tqdm


def Train(model, train_DL, val_DL, epoch, batch_size, max_len,
          criterion, optimizer, tokenizer, DEVICE, 
          save_history_path, save_model_path, scheduler = None):
    loss_history = {'train': [], 'val': []}
    best_loss = float('inf')
    for ep in range(epoch):
        model.train()
        train_loss = loss_epoch(model, train_DL, criterion, tokenizer, max_len, DEVICE, optimizer = optimizer, scheduler = scheduler)
        loss_history['train'] += [train_loss]

        model.eval()
        with torch.no_grad():
            val_loss = loss_epoch(model, val_DL, criterion, tokenizer, max_len, DEVICE)
            loss_history['val'] += [val_loss]
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({'model' : model,
                            'ep' : ep,
                            'optimizer': optimizer,
                            'scheduler' : scheduler}, save_model_path)
        print(f"Epoch {ep+1} : train loss : {train_loss:.5f}    val loss : {val_loss:.5f}   current_LR : {optimizer.param_groups[0]['lr']:.8f}")
        print('-' * 20)

    torch.save({'loss_history' : loss_history,
               'EPOCH' : epoch,
               'BATCH_SIZE' : batch_size}, save_history_path)

def Test(model, test_DL, criterion, tokenizer, max_len, DEVICE):
    model.eval()
    with torch.no_grad():
        test_loss = loss_epoch(model, test_DL, criterion, tokenizer, max_len, DEVICE)
    print(f'Test Loss : {test_loss:.3f} | Test PPL : {math.exp(test_loss):.3f}')

def loss_epoch(model, DL, criterion, tokenizer, max_len, DEVICE, optimizer = None, scheduler = None):
    N = len(DL.dataset)

    rloss = 0
    for src_texts, trg_texts in tqdm(DL, leave = False):
        src = tokenizer(src_texts, padding = True, truncation = True, max_length = max_len, return_tensors = 'pt', add_special_tokens = False).input_ids.to(DEVICE)
        trg_texts = ['</s> ' + s for s in trg_texts]
        trg = tokenizer(trg_texts, padding = True, truncation = True, max_length = max_len, return_tensors = 'pt').input_ids.to(DEVICE)

        y_hat = model(src, trg[:, :-1])[0]

        loss = criterion(y_hat.permute(0,2,1), trg[:, 1:])

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_b = loss.item() * src.shape[0]
        rloss += loss_b
    loss_e = rloss / N
    return loss_e

def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num

class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, LR_scale = 1):
        self.optimizer = optimizer
        self.current_step = 0
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.LR_scale = LR_scale

    def step(self):
        self.current_step += 1
        lrate = self.LR_scale * (self.d_model ** -0.5) * min(self.current_step ** -0.5, self.current_step * self.warmup_steps ** -1.5)
        self.optimizer.param_groups[0]['lr'] = lrate