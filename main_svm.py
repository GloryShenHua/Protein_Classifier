import os
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import label_binarize
from fea import feature_extraction
from sklearn.svm import SVC
from Bio.PDB import PDBParser
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class SVMModel:
# Todo
    """
        Initialize Support Vector Machine (SVM from sklearn) model.

        Parameters:
        - C (float): Regularization parameter. Default is 1.0.
        - kernel (str): Specifies the kernel type to be used in the algorithm. Default is 'rbf'.
    """

    """
        Train the Support Vector Machine model.

        Parameters:
        - train_data (array-like): Training data.
        - train_targets (array-like): Target values for the training data.
    """

    """
        Evaluate the performance of the Support Vector Machine model.

        Parameters:
        - data (array-like): Data to be evaluated.
        - targets (array-like): True target values corresponding to the data.

        Returns:
        - float: Accuracy score of the model on the given data.
    """

    def __init__(self, C=1.0, kernel='rbf'):
        self.model = SVC(C=C, kernel=kernel, max_iter=5000)
        self.scaler = StandardScaler()  # 初始化标准化器

    def train(self, train_data, train_targets):
        train_data = self.scaler.fit_transform(train_data)  # 对训练数据进行标准化
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        data = self.scaler.transform(data)  # 使用相同的缩放器对测试数据进行转换
        predictions = self.model.predict(data)
        return accuracy_score(targets, predictions)

class SVMFromScratch:
    def __init__(self, lr=0.001, num_iter=200, c=0.01):
        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = 0
        self.C = c
        self.mean = None
        self.std = None

    def compute_loss(self, y, predictions):
        """
        SVM Loss function:
        hinge_loss = 1/2 * ||w||^2 + C * sum(max(0, 1 - y * z))
        """
        num_samples = y.shape[0]
        # 1. Hinge Loss: max(0, 1 - y * z)
        hinge_loss = np.maximum(0, 1 - y * predictions)
        hinge_loss_sum = np.sum(hinge_loss)  # sum
        # 2. Regularization term: 1/2 * ||w||^2
        regularization = 0.5 * np.sum(self.weights ** 2)
        total_loss = regularization + self.C * hinge_loss_sum
        loss = total_loss / num_samples
        return loss
    
    def standardize(self, X):
        return (X - self.mean) / self.std

    #  todo:
    def train(self, train_data, train_targets):
        X = np.array(train_data)
        y = np.array(train_targets)

        # Convert tags to 1 and -1
        y = np.where(y == 0, -1, 1)

        # Standardize Data
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1e-3   
        X = self.standardize(X)  

        num_samples, num_features = X.shape
        # Initialize weights and biases
        self.weights = np.zeros(num_features)
        self.bias = 0   
        # Gradient descent updates parameters
        for iteration in range(self.num_iter):
            for i in range(num_samples):  
                pass
                # ### Todo
                # ### Calculate the output of a svm model
                svm_output = np.dot(X[i], self.weights) + self.bias 
                # ### Calculate the gradient dw, db  (提示：根据y[i]与svm_output是否符号一致，分类讨论dw，db)
                if y[i] * svm_output < 1:
                    dw = self.weights - self.C * y[i] * X[i]
                    db = -self.C * y[i]
                else:
                    dw = self.weights
                    db = 0
                
                # ### Update weights and bias
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            if iteration % 10 == 0:
                predictions = np.dot(X, self.weights) + self.bias
                loss = self.compute_loss(y, predictions)
                print(f"Iteration {iteration}, Loss: {loss}")
        

    def predict(self, X):
        # sign 
        X = self.standardize(X)  
        svm_model = np.dot(X, self.weights) + self.bias
        predictions = np.sign(svm_model)  
        return predictions

    def evaluate(self, data, targets):
        X = np.array(data)
        y = np.array(targets)
        y = np.where(y == 0, -1, 1)
        predictions = self.predict(X)
        return np.mean(predictions == y)
    

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
    
    ## Todo:Model Initialization 
    ## You can also consider other different settings
    #model = SVMModel(C=args.C,kernel=args.kernel)
    model = SVMFromScratch(lr=0.001, num_iter=200, c=0.01)


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
    parser = argparse.ArgumentParser(description="SVM Training and Evaluation")
    parser.add_argument('--C', type=float, default=1.0, help="Regularization parameter")
    parser.add_argument('--ent', action='store_true', help="Load data from a file using a feature engineering function feature_extraction() from fea.py")
    parser.add_argument('--kernel', type=str, default='rbf', help="Kernel type for SVM")
    args = parser.parse_args()
    main(args)

