import fasttext

model_pth = '/dat01/laizhiquan/psl/Project/UNMT/fastText/models/wmt16/en.bin'
test_pth = '/dat01/laizhiquan/psl/Project/UNMT/MUSE/results/translate/en-ro/europarl-v8.ro-en.en'

model = fasttext.load_model(model_pth)
words_in_emb = model.get_words()[0:200000]
total = 0
hit = 0
with open(test_pth, 'r', encoding='utf8') as f:
    lines = f.readlines()
    for line in lines:
        words_in_line = line.strip().split(' ')
        total += len(words_in_line)
        for word in words_in_line:
            word = word.strip('.?!:,()-')
            if word in words_in_emb:
                hit += 1
ratio = hit / total
print(ratio)