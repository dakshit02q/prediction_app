import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

def get_clean_data():
    data = pd.read_csv("data.csv")
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

    model1 = DecisionTreeClassifier(max_depth=5)
    model1.fit(xtrain, ytrain)

    model2 = SVC(kernel = 'linear' )
    model2.fit(xtrain, ytrain)

    #predicition 
    ypred = model.predict(xtest)
    print('accuracy of the model: ', accuracy_score(ytest, ypred))
    print('classification report:', classification_report(ytest, ypred))

    ypred1 = model1.predict(xtest)
    print('accuracy of the model1: ', accuracy_score(ytest, ypred1))
    print('classification report:', classification_report(ytest, ypred1))

    ypred2 = model2.predict(xtest)
    print('accuracy of the model2: ', accuracy_score(ytest, ypred2))
    print('classification report:', classification_report(ytest, ypred2))


    return model, model1, model2, scaler


def main():
    data = get_clean_data()

    model, model1, model2, scaler = create_model(data)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model,f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('model1.pkl', 'wb') as f:
        pickle.dump(model1, f)

    with open('model2.pkl', 'wb') as f:
        pickle.dump(model2, f)

if __name__ == '__main__':
    main()
