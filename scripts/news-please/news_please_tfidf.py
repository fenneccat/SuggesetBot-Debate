from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from pathlib import Path
import json
from datetime import datetime
from argparse import ArgumentParser

def get_timestamp():
    return datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

class Text:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.json_files = self.data_dir.glob('**/*.json')
        # print('# of files', len(list(self.json_files)))

    def __iter__(self):
        for f in self.json_files:
            with f.open(encoding='utf-8') as fin:
                obj = json.load(fin)
                try:
                    title = obj['title']
                    text = obj['text']
                    yield title.split() + text.split()
                except Exception as e:
                    # print(e)
                    yield []

if __name__ == '__main__':
    # Arguments
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=str)
    args = parser.parse_args()

    # run
    dataset = Text(args.data_dir)
    dct = Dictionary(dataset)
    print("# of documents", dct.num_docs)
    dataset = Text(args.data_dir)
    corpus = [dct.doc2bow(line) for line in dataset]
    model = TfidfModel(corpus)

    # Save
    output_dir = Path('./output')
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    timestamp = get_timestamp()
    fname = f'tfidf_{timestamp}'
    dct_path = output_dir / f'{fname}.dict'
    dct.save(str(dct_path))
    model_path = output_dir / f'{fname}.gensim'
    model.save(str(model_path))