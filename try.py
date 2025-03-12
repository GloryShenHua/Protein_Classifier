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
    """
    改进后的逻辑回归模型：
    - 使用 Pipeline 整合数据标准化和逻辑回归模型
    - 可选使用 GridSearchCV 自动调参
    """
    def __init__(self, use_grid_search=False):
        self.use_grid_search = use_grid_search
        
        # 定义流水线：先标准化再训练逻辑回归模型
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, multi_class='ovr'))
        ])
        
        # 定义超参数网格：这里只搜索 C 参数，可根据需要扩展
        self.param_grid = {
            'clf__C': [0.01, 0.1, 1, 10, 100],
            # 对于 lbfgs 求解器，仅支持 l2 惩罚；若用 liblinear 可加入 'l1'
        }
        self.best_model = None

    def train(self, train_data, train_targets):
        """
        训练模型：
        - 若 use_grid_search 为 True，则通过 GridSearchCV 自动调参
        - 否则直接训练流水线
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
    """
    从零开始实现的逻辑回归模型
    """
    def __init__(self):
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, train_data, train_targets, learning_rate=0.01, epochs=1000):
        num_samples, num_features = train_data.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # 这里仅适用于二分类问题
        for epoch in range(epochs):
            linear_model = np.dot(train_data, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)
            # 梯度下降更新参数
            dw = (1 / num_samples) * np.dot(train_data.T, (predictions - train_targets))
            db = (1 / num_samples) * np.sum(predictions - train_targets)
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def evaluate(self, data, targets):
        linear_model = np.dot(data, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        predictions = np.round(predictions)  # 将概率转换为 0 或 1
        return accuracy_score(targets, predictions)


def data_preprocess(args):
    """
    预处理数据：
    - 如果使用 --ent 参数，则从 feature_extraction() 获取特征，否则从预存文件加载特征
    - 从 cast 文件中读取每个任务的标签，每个任务要求如下：
        1 -> 训练正样本, 2 -> 训练负样本, 3 -> 测试正样本, 4 -> 测试负样本
      对于训练样本，将 1 映射为 1 (正样本)，2 映射为 0 (负样本)；
      对于测试样本，将 3 映射为 1 (正样本)，4 映射为 0 (负样本)；
    - 只有同时包含训练和测试样本的任务才会被保留
    """
    if args.ent:
        features, atom_names = feature_extraction()
    else:
        features = np.load('./data/diagrams.npy')
    
    # 读取 cast 文件（第一列为蛋白质标识，其余各列为不同任务的标签）
    cast = pd.read_table('./data/SCOP40mini_sequence_minidatabase_19.cast')
    cast.columns.values[0] = 'protein'
    
    data_list = []
    target_list = []
    
    # 遍历任务列（从第二列开始，共55个任务，假设每列包含的标签为 1,2,3,4）
    for task in range(1, 56):
        task_col = cast.iloc[:, task]
        # 筛选训练和测试样本（根据标签值）
        train_mask = (task_col == 1) | (task_col == 2)
        test_mask  = (task_col == 3) | (task_col == 4)
        
        # 检查训练和测试样本是否齐全
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(f"Warning: Task {task} 缺少训练或测试样本，跳过该任务。")
            continue
        
        # 根据 mask 过滤特征，注意 cast 的行顺序需与 features 顺序一致
        X_train = features[train_mask.values]
        X_test  = features[test_mask.values]
        
        # 将训练标签映射为二值：1 -> 正样本，2 -> 负样本
        y_train = task_col[train_mask].map({1: 1, 2: 0}).values
        # 测试标签映射为二值：3 -> 正样本，4 -> 负样本
        y_test = task_col[test_mask].map({3: 1, 4: 0}).values
        
        data_list.append((X_train, X_test))
        target_list.append((y_train, y_test))
    
    return data_list, target_list



def main(args):
    data_list, target_list = data_preprocess(args)
    
    if not data_list:
        print("没有足够的数据进行训练，请检查数据集的标签分布。")
        return

    task_acc_train = []
    task_acc_test = []

    # 这里选择使用 scikit-learn 实现的模型；如果想使用从零开始的实现，请替换为 LRFromScratch()
    model = LRModel()
    #model = LRFromScratch()
    
    for i in range(len(data_list)):
        train_data, test_data = data_list[i]
        train_targets, test_targets = target_list[i]

        print(f"Processing dataset {i+1}/{len(data_list)}")
        
        # 对于 LRFromScratch，仅适用于二分类问题；多分类任务请使用 LRModel 或自行扩展
        model.train(train_data, train_targets)
        train_accuracy = model.evaluate(train_data, train_targets)
        test_accuracy = model.evaluate(test_data, test_targets)

        print(f"Dataset {i+1}/{len(data_list)} - Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
        task_acc_train.append(train_accuracy)
        task_acc_test.append(test_accuracy)

    print("Training accuracy:", sum(task_acc_train) / len(task_acc_train))
    print("Testing accuracy:", sum(task_acc_test) / len(task_acc_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LR Training and Evaluation")
    parser.add_argument('--ent', action='store_true',
                        help="使用 feature_extraction() 从文件加载数据")
    args = parser.parse_args()
    main(args)
