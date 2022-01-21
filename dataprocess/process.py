import os
import subprocess

from utils.parser import get_parser

parser = get_parser()
option = parser.parse_args()

# more datasets can be found in http://www.statmt.org/wmt19/translation-task.html

onweb_path = 'http://data.statmt.org/news-commentary/v14/training/news-commentary-v14.de-en.tsv.gz'

ziped_path = os.path.join(option.sources_path, 'news_commentary_v14_de_en.gz')
unzip_path = os.path.join(option.sources_path, 'news_commentary_v14_de_en.tsv')

subprocess.run('mkdir -p %s' % option.sources_path, shell = True)

if not os.path.exists(ziped_path):
    os.system('wget %s -O %s' % (onweb_path, ziped_path))

if not os.path.exists(unzip_path):
    os.system('gzip -k -d %s' % (ziped_path))
    os.system('mv %s %s' % (ziped_path.split('.')[0], unzip_path))

data = []
with open(unzip_path, 'r', encoding = 'utf-8') as data_file:
    for line in data_file:
        if len(line.split('\t')) != 2:
            continue
        src_sent, trg_sent = line.split('\t')
        src_sent = src_sent.strip()
        trg_sent = trg_sent.strip()
        data.append((src_sent, trg_sent))

train_path = os.path.join(option.targets_path, option.train_file)
valid_path = os.path.join(option.targets_path, option.valid_file)
test_path  = os.path.join(option.targets_path, option.test_file )

subprocess.run('mkdir -p %s' % option.targets_path, shell = True)

train_radio = 0.9
valid_radio = 0.05
test_radio  = 0.05

total_data_len = len(data)
train_data_len = int(total_data_len * train_radio)
valid_data_len = int(total_data_len * valid_radio)
test_data_len  = int(total_data_len * test_radio )

start_idx = 0
train_end_idx = train_data_len
valid_end_idx = train_data_len + valid_data_len
test_end_idx  = train_data_len + valid_data_len + test_data_len

with open(train_path, 'w', encoding = 'utf-8') as train_file:
    for line in data[start_idx:train_end_idx]:
        train_file.write(line[0] + '\t' + line[1] + '\n')

with open(valid_path, 'w', encoding = 'utf-8') as valid_file:
    for line in data[train_end_idx:valid_end_idx]:
        valid_file.write(line[0] + '\t' + line[1] + '\n')

with open(test_path, 'w', encoding = 'utf-8') as test_file:
    for line in data[valid_end_idx:test_end_idx]:
        test_file.write(line[0] + '\t' + line[1] + '\n')
