import torch, os
for n in ['logiccbm_k312_250n_inception_v3.pt', 'logiccbm_k312_250n.pt',
          'cub_stage2_logic.pt', 'cub_e2e_logic.pt']:
    p = os.path.join('saved', n)
    if not os.path.exists(p):
        print('===', n, 'MISSING'); continue
    ck = torch.load(p, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck) if isinstance(ck, dict) else ck
    keys = list(sd.keys())
    prefixes = sorted(set(k.split('.')[0] for k in keys))
    meta = {k: ck.get(k) for k in ('acc', 'con_acc', 'head', 'n_neurons', 'joint')} if isinstance(ck, dict) else {}
    print('===', n)
    print('  prefixes:', prefixes)
    print('  sample keys:', keys[:4])
    print('  meta:', meta)
