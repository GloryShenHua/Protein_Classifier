import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from fea import feature_extraction
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


class LRModel:
    # todo:
    """
        Initialize Logistic Regression (from sklearn) model.

    """

    """
        Train the Logistic Regression model.

        Parameters:
        - train_data (array-like): Training data.
        - train_targets (array-like): Target values for the training data.
    """

    """
        Evaluate the performance of the Logistic Regression model.

        Parameters:
        - data (array-like): Data to be evaluated.
        - targets (array-like): True target values corresponding to the data.

        Returns:
        - float: Accuracy score of the model on the given data.
    """

    def __init__(self, use_grid_search=False):
        self.use_grid_search = use_grid_search
        
        #定义流水线：先标准化再训练逻辑回归模型
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, multi_class='ovr'))
        ])
        
        #定义超参数网格：这里只搜索C参数
        self.param_grid = {
            'clf__C': [0.01, 0.1, 1, 10, 100],
        }
        self.best_model = None

    def train(self, train_data, train_targets):
        """
        训练模型：
        -若use_grid_search为True,则通过GridSearchCV自动调参
        -否则直接训练流水线
        """
        if self.use_grid_search:
            grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=5, scoring='accuracy')
            grid_search.fit(train_data, train_targets)
            self.best_model = grid_search.best_estimator_
            print("Best parameters found:", grid_search.best_params_)
        else:
            self.pipeline.fit(train_data, train_targets)
            self.best_model = self.pipeline

    def evaluate(self, data, targets):
        """
        评估模型性能，返回准确率
        """
        predictions = self.best_model.predict(data)
        return accuracy_score(targets, predictions)


class LRFromScratch:
    # todo:
    def __init__(self):
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, train_data, train_targets, learning_rate=0.01, epochs=1000):
        num_samples, num_features = train_data.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        #仅适用于二分类问题
        for epoch in range(epochs):
            linear_model = np.dot(train_data, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)
            #梯度下降更新参数
            dw = (1 / num_samples) * np.dot(train_data.T, (predictions - train_targets))
            db = (1 / num_samples) * np.sum(predictions - train_targets)
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def evaluate(self, data, targets):
        linear_model = np.dot(data, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        predictions = np.round(predictions)  #将概率转换为0或1
        return accuracy_score(targets, predictions)


def data_preprocess(args):
    if args.ent:
        diagrams, atom_names = feature_extraction()
    else:
        diagrams = np.load('./data/diagrams.npy')
    cast = pd.read_table('./data/SCOP40mini_sequence_minidatabase_19.cast')
    cast.columns.values[0] = 'protein'

    data_list = []
    target_list = []
    for task in range(1, 56):  # Assuming only one task for now
        task_col = cast.iloc[:, task]
      
        ## todo: Try to load data/target
        #筛选训练和测试样本（根据标签值）
        train_mask = (task_col == 1) | (task_col == 2)
        test_mask  = (task_col == 3) | (task_col == 4)
        
        # 检查训练和测试样本是否齐全
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(f"Warning: Task {task} 缺少训练或测试样本，跳过该任务。")
            continue
        
        train_data = diagrams[train_mask.values]
        test_data  = diagrams[test_mask.values]
        
        #将训练标签映射为二值：1 -> 正样本，2 -> 负样本
        train_targets = task_col[train_mask].map({1: 1, 2: 0}).values
        #测试标签映射为二值：3 -> 正样本，4 -> 负样本
        test_targets = task_col[test_mask].map({3: 1, 4: 0}).values
        
        data_list.append((train_data, test_data))
        target_list.append((train_targets, test_targets))
    
    return data_list, target_list

def main(args):

    data_list, target_list = data_preprocess(args)

    task_acc_train = []
    task_acc_test = []
    

    #model = LRModel()
    model = LRFromScratch()
    for i in range(len(data_list)):
        train_data, test_data = data_list[i]
        train_targets, test_targets = target_list[i]

        print(f"Processing dataset {i+1}/{len(data_list)}")

        # Train the model
        model.train(train_data, train_targets)

        # Evaluate the model
        train_accuracy = model.evaluate(train_data, train_targets)
        test_accuracy = model.evaluate(test_data, test_targets)

        print(f"Dataset {i+1}/{len(data_list)} - Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        task_acc_train.append(train_accuracy)
        task_acc_test.append(test_accuracy)


    print("Training accuracy:", sum(task_acc_train)/len(task_acc_train))
    print("Testing accuracy:", sum(task_acc_test)/len(task_acc_test))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LR Training and Evaluation")
    parser.add_argument('--ent', action='store_true', help="Load data from a file using a feature engineering function feature_extraction() from fea.py")
    args = parser.parse_args()
    main(args)

