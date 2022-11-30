"""
References: https://github.com/auspicious3000/autovc/blob/master/solver_encoder.py
"""

from models import build_model
from losses import *
from meldataset import *
from utils import *
from torch.utils.tensorboard import SummaryWriter

global_epoch = 0

def main(config_path):
    global global_epoch

    # tensorboard
    if not os.path.exists('logs'):
        os.makedirs('logs')
        os.makedirs('logs/train')
        os.makedirs('logs/eval')

    writer_train = SummaryWriter(log_dir='logs/train')
    writer_eval = SummaryWriter(log_dir='logs/eval')

    # training configuration
    config = get_hparams_from_file(config_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config.train.seed)

    keys = ['loss_id', 'loss_id_psnt', 'loss_cd', 'loss_total']

    # load data
    train_list, val_list = get_data_list(config.data.training_files, config.data.validation_files)

    train_loader = build_dataloader(train_list,
                                    validation=False,
                                    batch_size=config.train.batch_size,
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

    global_step = global_epoch * len(train_loader)

    # grad scaler
    scaler = torch.cuda.amp.GradScaler() if (('cuda' in str(device)) and config.train.fp16_run) else None

    # lr scheduler

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.train.lr_decay, last_epoch=global_epoch-1)

    generator.train()

    while global_epoch < config.train.epochs:
        counter = 0
        for x_real, sid in train_loader:
            # train
            x_real, sid = x_real.to(device), sid.to(device) # [B, 1, 80, N], [B]

            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast(): # automatic mixed precision
                    x_identic, x_identic_psnt, code_real = generator(x_real.squeeze(1), sid, sid)
                    # reconstruction loss
                    loss_id = recon_loss(x_real, x_identic)
                    loss_id_psnt = recon_loss(x_real, x_identic_psnt)

                    # code semantic loss
                    code_reconst = generator(x_identic_psnt.squeeze(1), sid, None)
                    loss_cd = content_loss(code_real, code_reconst)

                    loss_total = loss_id + loss_id_psnt + config.train.lambda_cd * loss_cd
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                x_identic, x_identic_psnt, code_real = generator(x_real.squeeze(1), sid, sid)
                # reconstruction loss
                loss_id = recon_loss(x_real, x_identic)
                loss_id_psnt = recon_loss(x_real, x_identic_psnt)

                # code semantic loss
                code_reconst = generator(x_identic_psnt.squeeze(1), sid, None)
                loss_cd = content_loss(code_real, code_reconst)

                loss_total = loss_id + loss_id_psnt + config.train.lambda_cd * loss_cd

                loss_total.backward()
                optimizer.step()

            scheduler.step()

            writer_train.add_scalar('loss_id', loss_id, global_step)
            writer_train.add_scalar('loss_id_psnt', loss_id_psnt, global_step)
            writer_train.add_scalar('loss_cd', loss_cd, global_step)
            writer_train.add_scalar('loss_total', loss_total, global_step)

            # logging
            loss = {}
            loss['loss_id'] = loss_id.item()
            loss['loss_id_psnt'] = loss_id_psnt.item()
            loss['loss_cd'] = loss_cd.item()
            loss['loss_total'] = loss_total.item()

            if global_step % config.train.log_interval == 0:
                progress = counter / len(train_loader) * 100
                log = 'Epoch [{}/{}]: {:.2f}%, step {}'.format(global_epoch, config.train.epochs, progress, global_step)
                for key in keys:
                    log += ', {}: {:.4f}'.format(key, loss[key])
                print(log)

            # evaluate and save checkpoint
            if global_step % config.train.eval_interval == 0:
                generator.eval()
                with torch.no_grad():
                    for x_real_val, sid_val in val_loader:
                        x_real_val, sid_val = x_real_val.to(device), sid_val.to(device)

                        # remove else (pick the first sample of a batch)
                        x_real_val = x_real_val[:1]
                        sid_val = sid_val[:1]
                        break

                    x_identic_val, x_identic_psnt_val, code_real_val = generator(x_real_val.squeeze(1), sid_val, sid_val)
                    # reconstruction loss
                    loss_id_val = recon_loss(x_real_val, x_identic_val)
                    loss_id_psnt_val = recon_loss(x_real_val, x_identic_psnt_val)

                    # code semantic loss
                    code_reconst_val = generator(x_identic_psnt_val.squeeze(1), sid_val, None)
                    loss_cd_val = content_loss(code_real_val, code_reconst_val)

                    loss_total_val = loss_id_val + loss_id_psnt_val + config.train.lambda_cd * loss_cd_val

                    loss['loss_id'] = loss_id_val.item()
                    loss['loss_id_psnt'] = loss_id_psnt_val.item()
                    loss['loss_cd'] = loss_cd_val.item()
                    loss['loss_total'] = loss_total_val.item()

                    # logging
                    log = 'Validation at epoch {}, step {}'.format(global_epoch, global_step)
                    for key in keys:
                        log += ', {}: {:.4f}'.format(key, loss[key])
                    print(log)

                    # save checkpoint
                    save_checkpoint(config.train.save_path, generator, optimizer, config.train.learning_rate, global_epoch)

                generator.train()

                mel_gt = plot_spectrogram_to_numpy(x_real_val[0].squeeze(0).cpu().numpy())
                mel_gen = plot_spectrogram_to_numpy(x_identic_val[0].squeeze(0).cpu().numpy())
                mel_gen_psnt = plot_spectrogram_to_numpy(x_identic_psnt_val[0].squeeze(0).cpu().numpy())

                writer_eval.add_image('mel_gt', mel_gt, global_step, dataformats='HWC')
                writer_eval.add_image('mel_gen', mel_gen, global_step, dataformats='HWC')
                writer_eval.add_image('mel_gen_psnt', mel_gen_psnt, global_step, dataformats='HWC')

            global_step += 1
            counter += 1

        global_epoch += 1


if __name__ == '__main__':
    config_path = 'configs/config_autovc.json'
    main(config_path)

