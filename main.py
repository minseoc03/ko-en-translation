import torch
import random
import hydra
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from omegaconf import DictConfig
from hydra.utils import instantiate
from transformers import MarianTokenizer

from dataset import get_dataloader
from transformer import Transformer
from trainer import Train, NoamScheduler
from translation import translation

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg : DictConfig):
    #print current configuration setting
    print(cfg)

    #device setting
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    #define saved model path
    if cfg.trainer.pre_trained_use:
        save_model_path = 'pretrained/Transformer.pt'
        save_history_path = 'pretrained/Transformer_history.pt'
    else:
        save_model_path = 'Transformer_new.pt'
        save_history_path = 'Transformer_new_history.pt'

    #seed setting
    random_seed = cfg.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(random_seed)

    #declare tokenizer and receive its unique data
    tokenizer = MarianTokenizer.from_pretrained(cfg.tokenizer)
    eos_idx = tokenizer.eos_token_id
    pad_idx = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size   

    #create custom dataloader
    train_DL, val_DL, _ = get_dataloader(cfg.dataset)

    #declare model
    model = Transformer(vocab_size, pad_idx, DEVICE, cfg.model)

    #train model if not using pretrained model
    criterion = instantiate(cfg.trainer.criterion)
    criterion.ignore_index = pad_idx
    if cfg.trainer.pre_trained_use:
        params = [p for p in model.parameters() if p.requires_grad]
        if cfg.trainer.scheduler_name == 'Noam':
            optimizier = optim.Adam(params, lr = 0,
                                    betas = cfg.trainer.optimizers[0].betas, 
                                    eps = cfg.trainer.optimizers[0].eps,
                                    weight_decay = cfg.trainer.optimizers[0].weight_decay)
            scheduler = NoamScheduler(optimizer, 
                                      d_model = cfg.model.d_model, 
                                      warmup_steps = cfg.trainer.schedulers[0].warmup_steps, 
                                      LR_scale = cfg.trainer.schedulers[0].LR_scale)
        elif cfg.trainer.scheduler_name == 'Cos':
            optimizer = optim.Adam(params, lr=cfg.trainer.schedulers[1].LR_init,
                                   betas=cfg.trainer.optimizers[0].betas, 
                                   eps = cfg.trainer.optimizers[0].eps,
                                   weight_decay = cfg.trainer.optimizers[0].weight_decay)
            scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                                    cfg.trainer.scheduler[1].T0, 
                                                    cfg.trainer.scheduler[1].T_mult)

        Train(model, train_DL, val_DL, 
              cfg.trainer.epoch, cfg.dataset.batch_size, cfg.model.max_len,
              criterion, optimizer, tokenizer, DEVICE,
              save_history_path, save_model_path, scheduler)
    
    #load model pretrained model
    loaded = torch.load(save_model_path, map_location=DEVICE)
    load_model = loaded["model"]
    ep = loaded["ep"]
    optimizer = loaded["optimizer"]

    loaded = torch.load(save_history_path, map_location=DEVICE)
    loss_history = loaded["loss_history"]

    #inference
    translated_text, _ , _ , _ = translation(load_model, tokenizer, cfg.model.max_len, DEVICE, cfg.translation.src_text, atten_map_save = False)

    print(f"입력 : {cfg.translation.src_text}")
    print(f"번역 : {translated_text}")

if __name__ == "__main__":
    main()