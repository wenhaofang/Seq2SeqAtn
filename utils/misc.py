import os
import tqdm
import torch

from nltk.translate.bleu_score import corpus_bleu

def save_checkpoint(save_path, model, optim, epoch):
    checkpoint = {
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(load_path, model, optim):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])
    return checkpoint['epoch']

def save_sample(folder, reference_ids, hypothese_ids, reference_wds, hypothese_wds):
    reference_id_path = os.path.join(folder, 'reference_ids.txt')
    hypothese_id_path = os.path.join(folder, 'hypothese_ids.txt')
    reference_wd_path = os.path.join(folder, 'reference_wds.txt')
    hypothese_wd_path = os.path.join(folder, 'hypothese_wds.txt')
    for data, file_path in zip(
        [reference_ids, hypothese_ids, reference_wds, hypothese_wds],
        [reference_id_path, hypothese_id_path, reference_wd_path, hypothese_wd_path]
    ):
        with open(file_path, 'w', encoding = 'utf-8') as text_file:
            text_file.writelines([' '.join([str(item) for item in line]) + '\n' for line in data])

def train(module, loader, criterion, optimizer, device, clip):
    module.train()
    epoch_loss = 0
    for mini_batch in tqdm.tqdm(loader):
        mini_batch = [data_item.to(device) for data_item in mini_batch]
        source , source_length, target, _ = mini_batch
        output = module(source, source_length, target)
        loss = criterion(
            output[:, 1:].reshape(-1, output.shape[-1]),
            target[:, 1:].reshape(-1)
        )
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), clip)
        optimizer.step()
    return {
        'loss': epoch_loss / len(loader)
    }

def valid(module, loader, criterion, optimizer, device, vocab):
    module.eval()
    epoch_loss = 0
    reference_ids = []
    hypothese_ids = []
    reference_wds = []
    hypothese_wds = []
    with torch.no_grad():
        for mini_batch in tqdm.tqdm(loader):
            mini_batch = [data_item.to(device) for data_item in mini_batch]
            source , source_length, target, _ = mini_batch
            output = module(source, source_length, target, 0)
            loss = criterion(
                output[:, 1:].reshape(-1, output.shape[-1]),
                target[:, 1:].reshape(-1)
            )
            epoch_loss += loss.item()
            EOS_IDX = vocab['word2id'].get(vocab['special']['EOS_TOKEN'])
            for word_ids in target [:, 1:]:
                word_ids = word_ids.tolist()
                word_ids = word_ids[:word_ids.index(EOS_IDX) if EOS_IDX in word_ids else len(word_ids)]
                reference_ids.append([word_id for word_id in word_ids])
                reference_wds.append([vocab['id2word'].get(word_id) for word_id in word_ids])
            for word_ids in output [:, 1:].argmax(-1):
                word_ids = word_ids.tolist()
                word_ids = word_ids[:word_ids.index(EOS_IDX) if EOS_IDX in word_ids else len(word_ids)]
                hypothese_ids.append([word_id for word_id in word_ids])
                hypothese_wds.append([vocab['id2word'].get(word_id) for word_id in word_ids])
    bleu4 = corpus_bleu([[reference_id] for reference_id in reference_ids], hypothese_ids)
    bleu4 = round(bleu4, 4)
    return {
        'loss': epoch_loss / len(loader),
        'bleu': bleu4,
        'reference_ids': reference_ids,
        'hypothese_ids': hypothese_ids,
        'reference_wds': reference_wds,
        'hypothese_wds': hypothese_wds
    }
