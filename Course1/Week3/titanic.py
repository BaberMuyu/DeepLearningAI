from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import sklearn.preprocessing as preprocessing
from sklearn import model_selection
import pandas as pd
import numpy as np
import nn_model

### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    x = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


def get_dummies(df):
    dummies_Cabin = pd.get_dummies(df['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(df['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')

    df = pd.concat([df, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    return df


def scaling(df):
    scaler = preprocessing.StandardScaler()
    age_transpose = np.array(df['Age']).reshape(-1, 1)
    age_scale_param = scaler.fit(age_transpose)
    df['Age_scaled'] = scaler.fit_transform(age_transpose, age_scale_param)

    fare_transpose = np.array(df['Fare']).reshape(-1, 1)
    fare_scale_param = scaler.fit(fare_transpose)
    df['Fare_scaled'] = scaler.fit_transform(fare_transpose, fare_scale_param)
    return df


def preprocess(data_train, data_test):
    data_train, rfr = set_missing_ages(data_train)
    data_train = set_Cabin_type(data_train)
    data_train = get_dummies(data_train)
    data_train = scaling(data_train)

    data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0

    age_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])
    # 用得到的预测结果填补原缺失数据
    data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

    data_test = set_Cabin_type(data_test)
    data_test = get_dummies(data_test)
    data_test = scaling(data_test)

    return data_train, data_test


def load_titanic_data():
    data_train = pd.read_csv("data/train.csv")
    data_test = pd.read_csv("data/test.csv")

    data_train, data_test = preprocess(data_train, data_test)

    # 用正则取出我们要的属性值
    train_df = data_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.as_matrix()

    test_df = data_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    test_np = test_df.as_matrix()

    # Y即Survival结果
    train_Y = train_np[:, 0].reshape(1, -1)

    # X即特征属性值
    train_X = train_np[:, 1:].T
    s = len(train_X)
    train_X.reshape([s,-1])

    test_X = test_np.T
    s = len(test_X)
    test_X.reshape([s, -1])

    test_Y = data_test['PassengerId'].as_matrix()
    return train_X, train_Y, test_X, test_Y

if __name__ == "__main__":
    X, Y, TX, TY = load_titanic_data()
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    parameters = nn_model.nn_model(X, Y, 14, 20000, print_cost = False)
    predictions = nn_model.predict(parameters, X)
    print('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

    predictions = nn_model.predict(parameters, TX)
    result = pd.DataFrame({'PassengerId': TY, 'Survived': predictions.squeeze().astype(np.int32)})
    result.to_csv("data/logistic_regression_predictions.csv", index=False)


