import pickle
def analyze_sentiment(data):
    with open("vectorizer.pkl", 'rb') as fileobj2:
      vect = pickle.load(fileobj2)
    std_data = vect.transform(data)
    file1 = "trained_model.pkl"
    fileobj1 = open(file1, 'rb')
    model = pickle.load(fileobj1)

    # Make sure the number of features match
    if std_data.shape[1] != model.support_vectors_.shape[1]:
      raise ValueError(f"Number of features in input data ({std_data.shape[1]}) doesn't match the trained model ({model.support_vectors_.shape[1]})")
    
    prediction=model.predict(std_data)

    if (prediction[0] == 0):
      return 'Negative Rating'
    else:
      return 'Positive Rating'