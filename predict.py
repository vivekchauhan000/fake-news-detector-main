import pickle

# load model
model = pickle.load(open('../models/model.pkl', 'rb'))
vectorizer = pickle.load(open('../models/vectorizer.pkl', 'rb'))


def predict_news(text):
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]
    return "REAL" if result == 1 else "FAKE"


# test
if __name__ == "__main__":
    text = input("Enter news text: ")
    print(predict_news(text))
