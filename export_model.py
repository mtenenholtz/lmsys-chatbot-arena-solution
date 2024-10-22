import os
from transformers import AutoConfig, AutoModel

model = 'sfairXC/FsfairX-Gemma2-RM-v0.1'
exp_name = 'llm_pseudo_rm_orpo_500k_no_lstm_no_cap_rd_2'
path = f'/mnt/one/kaggle/lmsys-chatbot-arena/{model}-{exp_name}'

for fold in range(1):
    fold_path = f'{path}-fold-{fold}'
    print(fold_path)
    print(os.listdir(fold_path))
    checkpoint = os.listdir(fold_path)[0]

    if 'deberta' in model.lower():
        config = AutoConfig.from_pretrained(model)
        config.update({'max_position_embeddings': 1024})
        config.save_pretrained(f'{fold_path}/{checkpoint}')

    os.system(f'tar -czvf fold-{fold}.tar.gz --transform "s,^,fold-{fold}/," --exclude="optimizer.pt" -C {fold_path} {checkpoint}/')