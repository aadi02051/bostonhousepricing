import pickle, numpy as np

def predict_api():
    scalar = pickle.load(open('scaler.pkl','rb'))
    data = {'CRIM': 0.063,
        'ZN': 18,
        'INDUS': 2.31,
        'CHAS': 0,
        'NOX': 0.538,
        'RM': 6.575,
        'AGE': 65.2,
        'DIS': 4.09,
        'RAD': 1,
        'TAX': 296,
        'PTRATIO': 15.3,
        'B': 396.9,
        'LSTAT': 4.98}
    print(data)
    print(np.array(list(data.values())).reshape(1,-1).shape)
    print(scalar.transform(np.array(list(data.values())).reshape(1,-1)))
    #new_data = scalar.transform(np.array(list(data.values())).reshape(-1,1))

predict_api()
