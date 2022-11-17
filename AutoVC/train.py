"""
References: https://github.com/auspicious3000/autovc/blob/master/solver_encoder.py
"""

import torch
import torch.nn.functional as F
from models import Generator, build_model
from losses import *
from meldataset import *
from utils import *

global_epoch = 0

def main(config_path):
    global global_epoch

    # training configuration
    config = get_hparams_from_file(config_path)
    train_path = config.trian.training_files
    val_path = config.train.validation_files
    batch_size = config.train.batch_size
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    epochs = config.train.epochs
    log_interval = config.train.log_interval
    eval_interval = config.trian.eval_interval
    fp16_run = config.train.fp16_run
    lambda_cd = config.train.lambda_cd

    keys = ['loss_id', 'loss_id_psnt', 'loss_cd', 'loss_total']

    # load data
    train_list, val_list = get_data_list(train_path, val_path)
    train_loader = build_dataloader(train_list,
                                    validation=False,
                                    batch_size=batch_size,
                                    num_workers=4,
                                    device=device)
    val_loader = build_dataloader(val_list,
                                  validation=True,
                                  num_workers=2,
                                  device=device)

    # build model
    generator, optimizer = build_model(config)
    generator.to(device)

    # load checkpoint if there's any
    try:
        _, _, learning_rate, epoch = load_checkpoint(config.train.checkpoint_path,
                                                     generator, optimizer)
        global_epoch = epoch
    except:
        global_epoch = 0

    generator.train()

    for x_real, sid in train_loader:
        # train
        x_real, sid = x_real.to(device), sid.to(device)

        x_identic, x_identic_psnt, code_real = generator(x_real, sid, sid)
        # reconstruction loss
        loss_id = recon_loss(x_real, x_identic)
        loss_id_psnt = recon_loss(x_real, x_identic_psnt)

        # code semantic loss
        code_reconst = generator(x_identic_psnt, sid, None)
        loss_cd = content_loss(code_real, code_reconst)

        loss_total = loss_id + loss_id_psnt + lambda_cd * loss_cd
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # logging
        loss = {}
        loss['loss_id'] = loss_id.item()
        loss['loss_id_psnt'] = loss_id_psnt.item()
        loss['loss_cd'] = loss_cd.item()
        loss['loss_total'] = loss_total.item()

        if global_epoch % log_interval == 0:
            log = 'Epoch [{}/{}]'.format(global_epoch, epochs)
            for key in keys:
                log += ', {}: {:.4f}'.format(key, loss[key])
            print(log)

        # evaluate and save checkpoint
        if global_epoch % eval_interval == 0:
            generator.eval()
            with torch.no_grad():
                for x_real_val, sid_val in val_loader:
                    x_real_val, sid_val = x_real_val.to(device), sid_val.to(device)

                    # remove else (pick the first sample of a batch)
                    x_real_val = x_real_val[:1]
                    sid_val = sid_val[:1]
                    break

                x_identic_val, x_identic_psnt_val, code_real_val = generator(x_real_val, sid_val, sid_val)
                # reconstruction loss
                loss_id_val = recon_loss(x_real_val, x_identic_val)
                loss_id_psnt_val = recon_loss(x_real_val, x_identic_psnt_val)

                # code semantic loss
                code_reconst_val = generator(x_identic_psnt_val, sid_val, None)
                loss_cd_val = content_loss(code_real_val, code_reconst_val)

                loss_total_val = loss_id_val + loss_id_psnt_val + lambda_cd * loss_cd_val

                loss['loss_id'] = loss_id_val.item()
                loss['loss_id_psnt'] = loss_id_psnt_val.item()
                loss['loss_cd'] = loss_cd_val.item()
                loss['loss_total'] = loss_total_val.item()

                # logging
                log = 'Validation at epoch {}'.format(global_epoch)
                for key in keys:
                    log += ', {}: {:.4f}'.format(key, loss[key])
                print(log)

                # save checkpoint
                save_checkpoint(config.train.save_path, generator, optimizer, config.train.learning_rate, global_epoch)

            generator.train()

    global_epoch += 1



if __name__ == '__main__':
    x = torch.LongTensor([0, 1, 2, 3])

    emb = torch.nn.Embedding(4, 256)
    o = emb(x)
    print(o.shape)
