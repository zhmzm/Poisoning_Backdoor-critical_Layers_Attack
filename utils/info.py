import time
import math

def print_exp_details(args):
    info = information(args)
    for i in info:
        print(i)
    write_info(args, info)
    
def write_info_to_accfile(filename, args):
    info = information(args)
    f = open(filename, "w")
    for i in info:
        f.write(i)
        f.write('\n')
    f.close()    
    
def write_info(args, info):
    f = open("./"+args.save+'/'+"a_info.txt", "w")
    for i in info:
        f.write(i)
        f.write('\n')
    f.close()
    
def information(args):
    info = []
    info.append('======================================')
    info.append(f'    IID: {args.iid}')
    info.append(f'    Dataset: {args.dataset}')
    info.append(f'    Model: {args.model}')
    info.append(f'    Model Init: {args.init}')
    info.append(f'    Aggregation Function: {args.defence}')
    if math.isclose(args.malicious, 0) == False:
        info.append(f'    Attack method: {args.attack}')
        if 'adaptive' in args.attack:
            info.append(f'    Attack mode: {args.ada_mode}')
        info.append(f'    Attack tau: {args.tau}')
        info.append(f'    Fraction of malicious agents: {args.malicious*100}%')
        info.append(f'    Poison Frac: {args.poison_frac}')
        info.append(f'    Backdoor From {args.attack_goal} to {args.attack_label}')
        info.append(f'    Attack Begin: {args.attack_begin}')
        info.append(f'    Trigger Shape: {args.trigger}')
        if args.trigger == 'square' or args.trigger == 'pattern':
            info.append(f'    Trigger Position X: {args.triggerX}')
            info.append(f'    Trigger Position Y: {args.triggerY}')
        
    else:
        info.append(f'    -----No Attack-----')
        
    info.append(f'    Number of agents: {args.num_users}')
    info.append(f'    Fraction of agents each turn: {int(args.num_users*args.frac)}({args.frac*100}%)')
    info.append(f'    Local batch size: {args.local_bs}')
    info.append(f'    Local epoch: {args.local_ep}')
    info.append(f'    Client_LR: {args.lr}')
    # print(f'    Server_LR: {args.server_lr}')
    info.append(f'    Client_Momentum: {args.momentum}')
    info.append(f'    Global Rounds: {args.epochs}')
    if args.defence == 'RLR':
        info.append(f'    RobustLR_threshold: {args.robustLR_threshold}')
    elif args.defence == 'fltrust' or args.defence == 'fltrust_bn':
        info.append(f'    Dataset In Server: {args.server_dataset}')
    elif args.defence == 'flame' or args.defence == 'flame2':
        info.append(f'    Noise in FLAME: {args.noise}')
        if args.turn != 0:
            info.append('proportion of malicious are selected:'+str(args.wrong_mal/(int(args.malicious * max(int(args.frac * args.num_users), 1))*args.turn)))
            info.append('proportion of benign are selected:'+str(args.right_ben/((max(int(args.frac * args.num_users), 1) - int(args.malicious * max(int(args.frac * args.num_users), 1)))*args.turn)))
    elif args.defence == 'krum' or args.defence == 'multikrum':
        if args.turn != 0 and args.malicious != 0:
            p = args.wrong_mal/args.turn
            score_mal = args.mal_score/args.turn
            score_ben = args.ben_score/(args.turn*9)
            info.append('proportion of malicious are selected:'+str(args.wrong_mal/(int(args.malicious * max(int(args.frac * args.num_users), 1))*args.turn)))
            info.append(f'    Average score of malicious clients: {score_mal}')
            info.append(f'    Average score of benign clients: {score_ben}')
    info.append('======================================')
    return info

def get_base_info(args):
    if args.defence == 'RLR':
         base_info = '{}_{}_{}_{}_{}'.format(args.dataset,
                args.model, args.defence, args.robustLR_threshold, int(time.time()))
    else:
        base_info = '{}_{}_{}_{}'.format(args.dataset,
                    args.model, args.defence, int(time.time()))
    if math.isclose(args.malicious, 0) == False:
        base_info = base_info + '_{}_{}malicious_{}poisondata'.format(args.attack, args.malicious, args.poison_frac)
        if 'adaptive' in args.attack:
            base_info += '_mode{}'.format(args.ada_mode)
    else:
        base_info = base_info + '_no_malicious'
    return base_info