from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
pd.options.display.float_format = '{:.5f}'.format
import numpy as np
import faiss

data = pd.read_csv('data_01112021.csv')
FORMAT = '%(asctime)s:%(name)s:%(levelname)s - %(message)s'

logging.basicConfig(filename='knn.log',
                            filemode='w+',
                            format=FORMAT,
                            datefmt='%Y-%b-%d %X%z',
                            level=logging.INFO)

class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k
    #IndexFlatL2 is Euclidean distance
    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y
    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions
        
logging.info(data.corr()['isFraud'])
fraud_split_index = 4106 # total 8000
valid_split_index = 3177203 # total 6M
fraud_training_indexes = data[data['isFraud'] == 1].index[:fraud_split_index]
valid_training_indexes = data[data['isFraud'] == 0].index[:valid_split_index]
fraud_test_indexes = data[data['isFraud'] == 1].index[fraud_split_index:]
valid_test_indexes = data[data['isFraud'] == 0].index[valid_split_index:]
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X = pd.DataFrame(min_max_scaler.fit_transform(data), columns=data.columns, index=data.index).drop('isFraud', axis=1)
y = data['isFraud']
logging.info(X.head())
import numpy as np
faissModel = FaissKNeighbors(3)
for i in range(10):
    random_fraud_index = np.random.choice(fraud_training_indexes, 50)
    random_valid_index = np.random.choice(valid_training_indexes, 50000)
    train_subset = np.concatenate((random_fraud_index.astype(int), random_valid_index.astype(int)), axis=0)
    faissModelTrain = faissModel.fit(np.ascontiguousarray(X.iloc[train_subset.astype(int)]), np.ascontiguousarray( y.iloc[train_subset.astype(int)]))
test_subset = np.concatenate((fraud_test_indexes, valid_test_indexes), axis=0)
X_test = X.iloc[test_subset.astype(int)]
pred = faissModel.predict(np.ascontiguousarray(X_test))
from sklearn import metrics
# Model Accuracy: how often is the classifier correct?
logging.info("Accuracy: %f", metrics.accuracy_score(pred, y.iloc[test_subset]))
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
logging.info(confusion_matrix(pred,y.iloc[test_subset]))
logging.info(classification_report(pred, y.iloc[test_subset]))