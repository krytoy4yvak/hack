import sys
import re
import joblib


def input_one(input = None):
    file = open(input[1], 'r')
    lines = file.read()
    file.close()
    text = ' '.join(re.sub(r'[^a-zA-Zа-яА-Я ]', '', ' '.join(str(lines).lower().split())).split())
    print(text)
    model = joblib.load('models/validation/logreg_cv.joblib')
    cv = joblib.load('models/validation/count_vec.joblib')
    predict = model.predict(cv.transform([text]))
    return predict


if __name__ == '__main__':
    input_one(sys.argv)