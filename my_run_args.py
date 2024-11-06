import time
import sys
# import asteroid
import exp_models as M
from exp_data import sdr_improvement, neg_sdr
import torch
import wandb
# import pprint
from dataloaders import PSEData, collate_fn, Sampler, SamplerAll, SamplerFixNoise
from torch.utils.data import DataLoader
import os
import argparse
# from asteroid.losses.sdr import singlesrc_neg_snr




def run_train(spk_id, size, learning_rate, model_name, partition='120sec', load_ckpt=False, max_iter=None,
              min_epoch=0, max_epoch=200, is_rand_val=False, sampler_name=None, save_folder=None, batch_size=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = '/home/jb82/workspace_2024/GenDA_Challenge/Baseline/baseline_checkpoints'
    for f in os.listdir(checkpoint_path):
        if size in f:
            checkpoint_path = f'{checkpoint_path}/{f}'

    #loss_sdr = asteroid.losses.sdr.SingleSrcNegSDR('sisdr')
    #loss_sdr = singlesrc_neg_snr


    loss_sdr = neg_sdr

    # instantiates a new untrained model

    net, nparams, config = M.init_model(
        'convtasnet', size)

    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    if load_ckpt:
        # load weights from checkpoint
        print(torch.load(checkpoint_path, weights_only=True).keys())
        net.load_state_dict(
            torch.load(checkpoint_path).get('model_state_dict'),
            strict=True)
        optimizer.load_state_dict(
            torch.load(checkpoint_path).get('optimizer_state_dict')
        )
        # Refine learning rate
        for param_group in optimizer.param_groups:
            print(param_group['lr'], 'changed to', learning_rate)
            param_group['lr'] = learning_rate

    config['batch_size'] = batch_size
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
    save_path = os.path.join(save_folder, run_name)

    print(f"> run name: {run_name}")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    epoch_val_loss = 0
    val_sdr_improvement = 0

    with wandb.init(config=config, project='pse'):  # , entity='jsbae'):
        #wandb.config = config
        wandb.run.name = run_name

        # csv_path = f'/home/jb82/workspace_2024/GenDA_Challenge/Baseline/finetuned_checkpoints/csv_files/{partition}.csv'
        csv_path = f'/home/jb82/workspace_2024/GenDA_Challenge/Baseline/csv_files/{partition}.csv'

        if sampler_name == 'Sampler':
            batch_sampler = Sampler(csv_path, 'train')
            val_batch_sampler = Sampler(csv_path, 'val')
        elif sampler_name == 'SamplerAll':
            batch_sampler = SamplerAll(csv_path, 'train')
            val_batch_sampler = SamplerAll(csv_path, 'val')
        elif sampler_name == 'SamplerRandomNoiseCut':
            batch_sampler = Sampler(csv_path, 'train', is_random_noise_cut=True)
            val_batch_sampler = Sampler(csv_path, 'val', is_random_noise_cut=True)
        elif sampler_name == 'SamplerFixNoise':
            batch_sampler = SamplerFixNoise(csv_path, 'train', is_random_noise_cut=False)
            val_batch_sampler = SamplerFixNoise(csv_path, 'val', is_random_noise_cut=False)
        else:
            raise ValueError('Invalid Sampler Name')

        train_total_num = batch_sampler.get_data_len(spk_id, 'train')
        print('# # of train samples:', train_total_num)
    
        if not is_rand_val:
            # Load validation data
            # for fixed val samples and noises
            val_total_num = val_batch_sampler.get_data_len(spk_id, 'val')  # repeat 5 times
            print('# # of val samples:', val_total_num)
            val_batch_1 = val_batch_sampler.sample_batch(spk_id, val_total_num, 'val')
            val_batch_2 = val_batch_sampler.sample_batch(spk_id, val_total_num, 'val')
            val_batch_3 = val_batch_sampler.sample_batch(spk_id, val_total_num, 'val')
            VAL_batch_size = 8

            val_x = torch.cat([val_batch_1['x'], val_batch_2['x'], val_batch_3['x']], dim=0).to(device)
            val_t = torch.cat([val_batch_1['t'], val_batch_2['t'], val_batch_3['t']], dim=0).to(device)
            # val_x = val_batch['x'].to(device)
            # val_t = val_batch['t'].to(device)
            print(val_x.size())
            val_total_num *= 3
            
        wandb.watch(net, loss_sdr, log="all", log_freq=100)

        ep = 0
        prev_val_loss = 99999999
        no_improvement = 0
        total_steps = 0
        global_iter = 0

        # seen_mixtures = train_total_num  # was 45
        seen_mixtures = 45  # Fixed batch size. so the ep is not actually epoch. 
        print('# # of train samples:', batch_sampler.get_data_len(spk_id, 'train'))
        num_iter = (seen_mixtures // config['batch_size']) + 1 if (seen_mixtures % config['batch_size']) > 0 else (seen_mixtures // config['batch_size'])

        for _ in range(max_epoch):
            # TRAIN
            s_time = time.time()
            epoch_train_loss = 0
            epoch_sdr_improvement = 0
            net.train()
            for i in range(num_iter):
                batch = batch_sampler.sample_batch(spk_id, config['batch_size'], 'train')
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
                global_iter += 1

            epoch_train_loss /= num_iter
            epoch_sdr_improvement /= num_iter

            epoch_train_time = time.time() - s_time

            # VALIDATION
            if ep % 10 == 0:
                s_time = time.time()
                epoch_val_loss=0
                net.eval()
                
                if is_rand_val:
                    raise NotImplementedError
                    # Load validation data
                    # for fixed val samples and noises
                    val_total_num = val_batch_sampler.get_data_len(spk_id, 'val')
                    print('# # of val samples:', val_total_num)
                    val_batch = val_batch_sampler.sample_batch(spk_id, val_total_num, 'val')
                    VAL_batch_size = 1

                    val_x = val_batch['x'].to(device)
                    val_t = val_batch['t'].to(device)

                mini_steps = val_x.shape[0] // VAL_batch_size
                # assert mini_steps == val_total_num, f'{mini_steps} != {val_total_num}'
                for mini_batch_idx in range(mini_steps):
                    start = mini_batch_idx * VAL_batch_size
                    end = min(start + VAL_batch_size, val_x.shape[0])
                    x_mini = val_x[start:end]
                    t_mini = val_t[start:end]
                    y_mini = M.make_2d(net(x_mini)).detach()
                    loss = loss_sdr(y_mini, t_mini).mean().detach()
                    val_sdr_improvement += sdr_improvement(y_mini, t_mini, x_mini, reduction='mean')
                    epoch_val_loss += loss.data

                epoch_val_loss /= mini_steps
                val_sdr_improvement /= mini_steps
                epoch_val_time = time.time() - s_time
                #print(f"DEBUG {epoch_train_loss} {epoch_val_loss} {epoch_sdr_improvement}")

                #wandb.log({"train_loss": epoch_train_loss.data, "val_loss": epoch_val_loss.data,
                #           "train_sdr_improvement":epoch_sdr_improvement})
                wandb.log({"train_loss": epoch_train_loss.data, "val_loss": epoch_val_loss.data,
                        "train_sdr_improvement":epoch_sdr_improvement, "val_sdr_improvement":val_sdr_improvement}, step=ep)
                
                #if ep%10==0:
                print(f'> [Epoch]:{ep+1} [Train Loss]: {epoch_train_loss:.4f}, takes {epoch_train_time:.2f} sec')
                print(f'> [Epoch]:{ep+1} [Val Loss]: {float(epoch_val_loss):.4f}, takes {epoch_val_time:.2f} sec')

                if ep in [100, 200]:
                    print(f'# Save for {ep} ckpt')
                    ckpt_name = f"{save_path}/model_{ep+1}.ckpt"
                    ckpt = {}
                    ckpt['model_state_dict'] = net.state_dict()
                    ckpt['optim_state_dict'] = optimizer.state_dict()
                    ckpt['epoch'] = ep
                    ckpt['config'] = config
                    torch.save(ckpt, ckpt_name)

                if epoch_val_loss < prev_val_loss :
                    prev_val_loss = epoch_val_loss

                    # Save best checkpoint
                    print('# Save for best ckpt')
                    best_ckpt_name = f"{save_path}/model_best.ckpt"
                    best_ckpt = {}
                    best_ckpt['model_state_dict'] = net.state_dict()
                    best_ckpt['optim_state_dict'] = optimizer.state_dict()
                    best_ckpt['epoch'] = ep
                    best_ckpt['config'] = config
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
                
                if no_improvement >= 3 and ep > min_epoch:
                    print(f"> Training finished [epochs]:{ep+1} [steps]: {total_steps}")
                    # Save best checkpoint first
                    torch.save(best_ckpt, best_ckpt_name)

                    ckpt_name = f"{save_path}/model_last_{ep+1}.ckpt"
                    ckpt = {}
                    ckpt['model_state_dict'] = net.state_dict()
                    ckpt['optim_state_dict'] = optimizer.state_dict()
                    ckpt['epoch'] = ep
                    ckpt['config'] = config
                    torch.save(ckpt, ckpt_name)
                    break
            ep+=1
        # Save best checkpoint before leaving
        torch.save(best_ckpt, best_ckpt_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--speaker_id", type=str, required=True)
    parser.add_argument("-r", "--learning_rate", type=float, required=True)
    parser.add_argument("-i", "--size", type=str, required=True)
    parser.add_argument("-p", "--partition", type=str, required=True)
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument("-ex", "--experiment", type=str, required=True)
    args = parser.parse_args()

    if args.experiment == 'fixnosie_200':
        min_epoch = 0
        max_epoch = 201
        torch.cuda.empty_cache()
        is_rand_val = False
        sampler_name = 'SamplerFixNoise'
        save_folder = '/home/jb82/workspace_2024/GenDA_Challenge/Baseline/results/finetuned_checkpoints_-5to5_fixnoise_randomcut'
        batch_size = 8
    elif args.experiment == 'fixnoise_1000':
        min_epoch = 0
        max_epoch = 1001
        torch.cuda.empty_cache()
        is_rand_val = False
        sampler_name = 'SamplerFixNoise'
        save_folder = '/home/jb82/workspace_2024/GenDA_Challenge/Baseline/results/finetuned_checkpoints_-5to5_fixnoise_randomcut_1000'
        batch_size = 8
    elif args.experiment == 'fixnoise_200_virtual':
        min_epoch = 0
        max_epoch = 201
        torch.cuda.empty_cache()
        is_rand_val = False
        sampler_name = 'SamplerFixNoise'
        save_folder = '/home/jb82/workspace_2024/GenDA_Challenge/Baseline/results/finetuned_checkpoints_-5to5_fixnoise_randomcut_virtual'
        batch_size = 8
    else:
        raise ValueError('Invalid Experiment Name')

    print('#########################3')
    print('# SAMPLER:', sampler_name)
    print('# save_path:', save_folder)
    print('# Batch Size:', batch_size)
    print('#########################')

    run_train(spk_id=args.speaker_id, size=args.size, 
              learning_rate=args.learning_rate, 
              model_name=args.model_name,
              partition=args.partition,
              load_ckpt=True, max_iter=None,
              min_epoch=min_epoch,
              max_epoch=max_epoch,
              is_rand_val=is_rand_val,
              sampler_name=sampler_name,
              save_folder=save_folder,
              batch_size=batch_size)