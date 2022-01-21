import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # For Basic
    parser.add_argument('--name', default = 'main', help = '')

    # For Loader
    parser.add_argument('--sources_path', default = 'datasources', help = '')
    parser.add_argument('--targets_path', default = 'datatargets', help = '')

    parser.add_argument('--train_file', default = 'train_data.tsv', help = '')
    parser.add_argument('--valid_file', default = 'valid_data.tsv', help = '')
    parser.add_argument('--test_file' , default = 'test_data.tsv' , help = '')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    option = parser.parse_args()
