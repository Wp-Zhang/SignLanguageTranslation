import sys
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk import word_tokenize
if __name__ == "__main__":
    pred_path = sys.argv[1]
    data_path = sys.argv[2]
    with open(pred_path, "r") as file:
        pred = file.readlines()

    with open(data_path, "r") as file:
        target = file.readlines()
    
    scores = [nltk.meteor([word_tokenize(t.lower())], word_tokenize(p.lower())) for t,p in zip(target, pred)]
    print(sum(scores)/len(scores))
