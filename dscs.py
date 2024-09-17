import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def get_clean_data():
    data = pd.read_csv(r"Cancer_Data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis = 1)
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    print(data.head())
    return data


def create_model(data):
    x = data.drop(['diagnosis'], axis = 1)
    y = data['diagnosis']
    scaler = StandardScaler()
    x_scale = scaler.fit_transform(x)
    #split the data 
    xtrain, xtest, ytrain, ytest = train_test_split(x_scale, y, test_size = 0.2, random_state = 42)

    #trainig the data 
    model = LogisticRegression()
    model.fit(xtrain, ytrain)

    #predicition 
    ypred = model.predict(xtest)
    print('accuracy of the model: ', accuracy_score(ytest, ypred))
    print('classification report:', classification_report(ytest, ypred))

    return model, scaler


def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model,f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    main()
