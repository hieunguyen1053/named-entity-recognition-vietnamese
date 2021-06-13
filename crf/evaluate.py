from tqdm import tqdm

from crf import CRF_NER
from feature_extractor import FeatureExtractor


def _get_tags(sents):
    tags = []
    for sent_idx, iob_tags in enumerate(sents):
        curr_tag = {'type': None, 'start_idx': None,
                    'end_idx': None, 'sent_idx': None}
        for i, tag in enumerate(iob_tags):
            if tag == 'O' and curr_tag['type']:
                tags.append(tuple(curr_tag.values()))
                curr_tag = {'type': None, 'start_idx': None,
                            'end_idx': None, 'sent_idx': None}
            elif tag.startswith('B'):
                curr_tag['type'] = tag[2:]
                curr_tag['start_idx'] = i
                curr_tag['end_idx'] = i
                curr_tag['sent_idx'] = sent_idx
            elif tag.startswith('I'):
                curr_tag['end_idx'] = i
        if curr_tag['type']:
            tags.append(tuple(curr_tag.values()))
    tags = set(tags)
    return tags


def f_measure(y_true, y_pred):
    tags_true = _get_tags(y_true)
    tags_pred = _get_tags(y_pred)

    ne_ref = len(tags_true)
    ne_true = len(set(tags_true).intersection(tags_pred))
    ne_sys = len(tags_pred)
    if ne_ref == 0 or ne_true == 0 or ne_sys == 0:
        return 0
    p = ne_true / ne_sys
    r = ne_true / ne_ref
    f1 = (2 * p * r) / (p + r)

    return f1


def evaluate(model, filepath):
    data = open(filepath, 'r', encoding='utf-8').read()
    sents = data.split('\n\n')

    test_data = []
    for sent in sents:
        x = []
        items = sent.split('\n')
        for item in items:
            word, _, _, tag = item.split('\t')
            x.append((word, tag))
        test_data.append(x)

    feature_extractor = FeatureExtractor()
    X_test, y_test = feature_extractor.extract(test_data)
    y_pred = model.predict(X_test)
    return f_measure(y_test, y_pred)


if __name__ == '__main__':
    model = CRF_NER.load('/Users/hieunguyen/Desktop/NLP/Master/NER/crf/model.crfsuite')
    evaluate(model, '/Users/hieunguyen/Desktop/NLP/Master/NER/data/vlsp2016/test.txt')
