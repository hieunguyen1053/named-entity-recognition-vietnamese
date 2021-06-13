from feature_extractor import FeatureExtractor
from crf import CRF_NER

if __name__ == '__main__':
    data = open('/Users/hieunguyen/Desktop/NLP/Master/NER/data/vlsp2016/train.txt', 'r', encoding='utf-8').read()
    sents = data.split('\n\n')
    data = open('/Users/hieunguyen/Desktop/NLP/Master/NER/data/vlsp2016/dev.txt', 'r', encoding='utf-8').read()
    sents += data.split('\n\n')

    train_data = []
    for sent in sents:
        x = []
        items = sent.split('\n')
        for item in items:
            word, _, _, tag = item.split('\t')
            x.append((word, tag))
        train_data.append(x)

    feature_extractor = FeatureExtractor()
    X_train, y_train = feature_extractor.extract(train_data)

    model = CRF_NER(
        c1=1.0,
        c2=1e-3,
        max_iterations=200,
        all_possible_transitions=True,
        verbose=True,
    )

    model.fit(X_train, y_train)
    model.save("model.crfsuite")