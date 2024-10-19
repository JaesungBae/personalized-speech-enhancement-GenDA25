import sys
# import asteroid
import exp_models as M
from exp_data import sdr_improvement, neg_sdr
import torch
import wandb
# import pprint
from dataloaders import PSEData, collate_fn, Sampler
from torch.utils.data import DataLoader
import os
import argparse
# from asteroid.losses.sdr import singlesrc_neg_snr


MAX_EPOCH = 1000
torch.cuda.empty_cache()


def run_train(spk_id, size, learning_rate, model_name, partition='120sec', load_ckpt=False, max_iter=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = '/home/jb82/workspace_2024/GenDA_Challenge/Baseline/baseline_checkpoints'
    for f in os.listdir(checkpoint_path):
        if size in f:
            checkpoint_path = f'{checkpoint_path}/{f}'

    save_path = '/home/jb82/workspace_2024/GenDA_Challenge/Baseline/finetuned_checkpoints'

    #loss_sdr = asteroid.losses.sdr.SingleSrcNegSDR('sisdr')
    #loss_sdr = singlesrc_neg_snr
    loss_sdr = neg_sdr

    # instantiates a new untrained model
    net, nparams, config = M.init_model(
        'convtasnet', size)
    if load_ckpt:
        # load weights from checkpoint
        print(torch.load(checkpoint_path, weights_only=True).keys())
        net.load_state_dict(
            torch.load(checkpoint_path).get('model_state_dict'),
            strict=True)
    net = net.to(device)

    config['batch_size'] = 4
    config['loss_type'] = 'neg_sisdr'
    config['fine_tune']='synthesized'
    config['seconds'] = {'train':60, 'val':30}
    config['spk_id'] = spk_id
    config['learning_rate'] = learning_rate
    config['partition'] = partition

    ####For checkpoint saving
    if load_ckpt:
        run_name = f"new_{model_name}_spk{spk_id}_{size}_{learning_rate}_{partition}"
    else:
        run_name = f"new_{model_name}_spk{spk_id}_{size}_{learning_rate}_no_init"
    save_path = os.path.join(save_path, run_name)

    print(f"> run name: {run_name}")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with wandb.init(config=config, project='pse'):  # , entity='jsbae'):
        #wandb.config = config
        wandb.run.name = run_name

        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        if model_name=='specialist':
            csv_path = f'/home/jb82/workspace_2024/GenDA_Challenge/Baseline/finetuned_checkpoints/csv_files/{partition}.csv'
            batch_sampler = Sampler(csv_path, 'train')
            val_batch_sampler = Sampler(csv_path, 'val')
            
        # if partition=='audiolm':
        #     val_path = f"/home/anakuzne/data/AudioLM_subset/val/spk_{spk_id}.pt"
        # elif partition=='audiolm_10spk':
        #     val_path = f'/home/anakuzne/data/AudioLM_10spk/val/spk_{spk_id}.pt'
        # elif partition=='yourtts_28sec_set2':
        #     val_path = f"/home/anakuzne/data/YourTTS_28sec_set2/val/spk_{spk_id}.pt"
        # elif partition=='yourtts_60sec_set2':
        #     val_path = f"/home/anakuzne/data/YourTTS_60sec_set2/val/spk_{spk_id}.pt"
        # elif partition=='yourtts_120sec_set1':
        #     val_path = f"/home/anakuzne/data/YourTTS_120sec_set1/val/spk_{spk_id}.pt"
        # elif partition=='yourtts_60sec_set1':
        #     val_path = f"/home/anakuzne/data/YourTTS_60sec_set1/val/spk_{spk_id}.pt"
        # elif partition=='yourtts_30sec_10spk':
        #     val_path = f"/home/anakuzne/data/YourTTS_30sec_10spk/val/spk_{spk_id}.pt"
        # elif partition=='yourtts_60sec_10spk':
        #     val_path = f"/home/anakuzne/data/YourTTS_60sec_10spk/val/spk_{spk_id}.pt"
        # elif partition == 'xtts_v2_40utt':
        #     val_path = f''

        # else:
        #     assert False
        #     val_path = f"/home/anakuzne/data/LibriTestSynth/val/spk_{spk_id}.pt"
        # val_batch = torch.load(val_path)
    
        wandb.watch(net, loss_sdr, log="all", log_freq=100)

        ep = 0
        prev_val_loss = 99999999
        no_improvement = 0
        total_steps = 0
        #num_iter = (1000 // config['batch_size']) + 1 if (1000 % config['batch_size']) > 0 else (1000 // config['batch_size'])

        seen_mixtures = 45
        print('# # of train samples:', len(batch_sampler))
        num_iter = (seen_mixtures // config['batch_size']) + 1 if (seen_mixtures % config['batch_size']) > 0 else (seen_mixtures // config['batch_size'])

        for _ in range(MAX_EPOCH):
            epoch_train_loss = 0
            epoch_sdr_improvement = 0
            net.train()
            for i in range(num_iter):
                batch = batch_sampler.sample_batch(int(spk_id), config['batch_size'], 'train')
                x = batch['x'].to(device)
                #print(f'DEBUG {x.shape}')
                t = batch['t'].to(device)
                y = M.make_2d(net(x))
                loss = loss_sdr(y, t).mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                epoch_train_loss+=loss.data
                epoch_sdr_improvement += sdr_improvement(y, t, x, reduction='mean')
                total_steps+=config['batch_size']

            #print(f"Before div: {epoch_sdr_improvement}")
            epoch_train_loss /= num_iter
            epoch_sdr_improvement /= num_iter
            
            #print(f"After div: {epoch_sdr_improvement}")

            #Validation loop
            epoch_val_loss=0
            net.eval()

            VAL_TOTAL_NUM = 5
            VAL_BATCH_SIZE = 1
        
            val_batch = batch_sampler.sample_batch(int(spk_id), VAL_TOTAL_NUM, 'val')
            x = val_batch['x'].to(device)
            t = val_batch['t'].to(device)

            epoch_val_loss = 0
            val_sdr_improvement = 0

            mini_steps = x.shape[0] // VAL_BATCH_SIZE
            for mini_batch_idx in range(mini_steps):
                start = mini_batch_idx * VAL_BATCH_SIZE
                end = min(start + VAL_BATCH_SIZE, x.shape[0])
                x_mini = x[start:end]
                t_mini = t[start:end]
                y_mini = M.make_2d(net(x_mini))
                loss = loss_sdr(y_mini, t_mini).mean()
                val_sdr_improvement += sdr_improvement(y_mini, t_mini, x_mini, reduction='mean')
                epoch_val_loss += loss.data

            epoch_val_loss /= VAL_TOTAL_NUM
            val_sdr_improvement /= VAL_TOTAL_NUM
            #print(f"DEBUG {epoch_train_loss} {epoch_val_loss} {epoch_sdr_improvement}")

            #wandb.log({"train_loss": epoch_train_loss.data, "val_loss": epoch_val_loss.data,
            #           "train_sdr_improvement":epoch_sdr_improvement})
            wandb.log({"train_loss": epoch_train_loss.data, "val_loss": epoch_val_loss.data,
                       "train_sdr_improvement":epoch_sdr_improvement, "val_sdr_improvement":val_sdr_improvement})
            
            #if ep%10==0:
            print(f'> [Epoch]:{ep+1} [Train Loss]: {epoch_train_loss:.4f}')
            print(f'> [Epoch]:{ep+1} [Val Loss]: {float(epoch_val_loss):.4f}')

            if epoch_val_loss < prev_val_loss :
                prev_val_loss = epoch_val_loss

                # Save best checkpoint
                print('# Save for best ckpt')
                ckpt_name = f"{save_path}/model_best.ckpt"
                ckpt = {}
                ckpt['model_state_dict'] = net.state_dict()
                ckpt['optim_state_dict'] = optimizer.state_dict()
                ckpt['epoch'] = ep
                ckpt['config'] = config
                torch.save(ckpt, ckpt_name)
                no_improvement = 0
            #elif epoch_val_loss > prev_val_loss:
            #    #Reduce lr by half
            #    min_learning_rate = 1e-5
            #    for g in optimizer.param_groups:
            #        g['lr'] = max(min_learning_rate, g['lr']/2)
            #        new_lr = g['lr']
            #    print(f'Reduced LR to {new_lr}')
            else:
                # no_improvement += (num_iter*config['batch_size']) + 1 if (seen_mixtures % config['batch_size']) > 0 else (seen_mixtures // config['batch_size'])
                no_improvement += 1
            
            if no_improvement >= 50:
                print(f"> Training finished [epochs]:{ep+1} [steps]: {total_steps}")
                ckpt_name = f"{save_path}/model_last_{ep+1}.ckpt"
                ckpt = {}
                ckpt['model_state_dict'] = net.state_dict()
                ckpt['optim_state_dict'] = optimizer.state_dict()
                ckpt['epoch'] = ep
                ckpt['config'] = config
                torch.save(ckpt, ckpt_name)
                break
            ep+=1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--speaker_id", type=str, required=True)
    parser.add_argument("-r", "--learning_rate", type=float, required=True)
    parser.add_argument("-i", "--size", type=str, required=True)
    parser.add_argument("-p", "--partition", type=str, required=True)
    args = parser.parse_args()

    run_train(spk_id=args.speaker_id, size=args.size, 
              learning_rate=args.learning_rate, 
              model_name='specialist', 
              partition=args.partition,
              load_ckpt=True, max_iter=None)