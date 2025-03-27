import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

def one_hot_encoder(X, onehot):
    # spares_output=False will returns an narray
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    encoded = encoder.fit_transform(X[onehot])

    X = X.drop(columns=onehot).values
    X = np.column_stack([X, encoded])
    return X

# Creates new subsets from a dataset
def create_datasets(df, X_labels, y_label, onehot=None, size=None):
    datasets = list()
    if size == None:
        size = df.shape[0]

    for idx in range(len(y_label)):
        y = df[y_label[idx]]
        X = df[X_labels[idx]]
        
        if onehot == None: 
            print("True")
            # One Hot Encode all categorical variables
            X = pd.get_dummies(X, drop_first=True)
            X = X.values
        else:
            # One Hot Encode specific categorical variables
            X = one_hot_encoder(X, onehot[idx])

        datasets.append(  {"X":X[:size], "y":y.iloc[:size].values} )
                  
    return datasets


class ModelValidator:
    def __init__(self, datasets):
        self.datatsets = datasets  
        self.Xy_test_train_validate = list()
        #self.scaler = StandardScaler()
        self.verbose = False # Set this to True to print out information during tuning
        self.score_tuning = "accuracy" # The score type used during tuining
        self.score_evaluation = "accuracy" # The score type used during evaluation
        self.cv = 5
        self.voting = "hard"
        self.scaler = MinMaxScaler()
        self.normalizer = Normalizer()
        self.model_results_validation = list()
        self.model_results_test = list()
        

    def split(self, X, y):
        # tr = Train 
        # va = Validation (used during model tuning)
        # te = Test (used during evaluation) 
        Xtr, Xte, ytr, yte  =  train_test_split(X, y, test_size=.3)
        Xte, Xva, yte, yva  =  train_test_split(Xte, yte, test_size=.5)
        self.Xy_test_train_validate.append([Xtr, Xte, Xva, ytr, yte, yva]) 
    
    # These three methods exists in the sklearn library, 
    # but they were recrated to make them a more
    # clear and there implimitation more consistant
    def recall(self, TP, FN):
        return TP / (TP+FN)

    def precision(self, TP, FP):
        return TP / (TP+FP)
    
    def accuracy(self, TP,TN,FP,FN):
        return (TP+TN) / (TP+TN+FP+FN)
    
    def calculate_scores(self, y, y_hat, verbose=False):
        cm = confusion_matrix(y, y_hat) 
        recall = self.recall(cm[1,1], cm[1,0]) 
        precision = self.precision(cm[1,1], cm[0,1]) 
        accuracy = self.accuracy(cm[1,1], cm[0,0], cm[0,1], cm[1,0]) 
       
        if verbose:
            print( f"{recall=}"    )
            print( f"{precision=}" )
            print( f"{accuracy=}", end="\n\n"  )
        
        return recall, precision, accuracy    


    # Use max to pick the model dataset combo with highest score 
    def find_best_combination(self):
        return max(self.model_results_test, key=lambda di: di[self.score_evaluation])
    

    def initiat_model_tuning(self):
        # Loops through the datasets
        for ds_idx, dataset in enumerate(self.datatsets):
            self.split(dataset["X"], dataset["y"])

            # Loops through and tunes the model for each dataset
            for model in ["gradient_boosting", "decision_tree", "knn", "logistic_regression", "multinomial_nb"]:  
                self.execute_model_tuning(model, ds_idx)
            
            # Trains and addes the voting classifier
            self.voting_classifier(ds_idx)
        
        # During tuining the models has been evaluate with validation data
        # in this step all models will be evaluated with training data
        self.evaluate_models()


    def execute_model_tuning(self, model, ds_idx):

        Xtr, Xte, Xva, ytr, yte, yva = self.Xy_test_train_validate[ds_idx]

        # The models available to use
        models={
            "decision_tree":DecisionTreeClassifier,
            "gradient_boosting":GradientBoostingClassifier,
            "knn":KNeighborsClassifier,
            "logistic_regression":LogisticRegression, 
            "multinomial_nb":MultinomialNB, 
        }

        # The hyper parameters was taken from an LLM 
        # LLM: Gemini 2.0 Flas 
        # Prompt: How do I implement <model> in a GridSearchCV
        param_grid={
            "gradient_boosting":{
                "gradient_boosting__n_estimators": [100, 200, 300],
                "gradient_boosting__learning_rate": [0.01, 0.1, 0.2],

            },

            "decision_tree":{
                "decision_tree__criterion": ["gini", "entropy"],
                "decision_tree__max_depth": [3,6,9,12,15],
                "decision_tree__min_samples_split": [2,4,8,10],
                "decision_tree__min_samples_leaf": [1,3,6],
                "decision_tree__max_features": [ "sqrt", "log2", None],
            },

            "knn":{
                "knn__n_neighbors": [1, 2, 3, 5, 8, 13]
            },

            "logistic_regression":{
                 "logistic_regression__C": [0.001, 0.01, 0.1, 1, 10, 100],
                "logistic_regression__solver": ["liblinear", "saga"],
                "logistic_regression__max_iter": [100, 200, 300]
            },

            "multinomial_nb":{
                "multinomial_nb__alpha": [0.1, 0.5, 1.0, 2.0],
                "multinomial_nb__fit_prior": [True, False]
         },}

        # First scales, then normalizes, then retrives the model(?)
        # Note that the scaling and normalization is baked in to the model 
        # and will take effekt during both .fit and .predict so test validation
        # and train don't neeedt to be be scaled and normalized before they 
        # are used as argument in the model. 
        pipe = Pipeline([
            ("scaler", self.scaler),
            ("normalizeer", self.normalizer), 
            (model, models[model]())
        ]) 

        # Sets up the Grid Search
        # Note that cv, scoring and verbose can be changed 
        # by changing the class attributes
        classifier = GridSearchCV(
            estimator=pipe, 
            param_grid=param_grid[model], 
            cv=self.cv, # It could be worth trying 10
            scoring=self.score_tuning,
            n_jobs=-1,
            verbose=self.verbose
        )
        
        if self.verbose: 
            print(f"{model} : dataset {ds_idx}") 
        
        classifier.fit(Xtr, ytr)           
        best_version = classifier.best_estimator_
        y_hat = best_version.predict(Xva)

        # Calculates the scores
        precision, recall, accuracy = self.calculate_scores(yva, y_hat, self.verbose)

        # Stores all the information about the best model in a list.dict
        # The best model/datset pair can now easily be determined with max()
        # The best model can be evaluated dynamically using only the information this list + Xtr and yte
        self.model_results_validation.append({
            "dataset":ds_idx, 
            "model_name":model,  
            "classifier":best_version,
            "recall":recall, 
            "precision":precision, 
            "accuracy":accuracy,
        })


    def voting_classifier(self, ds_idx):
        
        # Splits the data
        Xtr, Xte, Xva, ytr, yte, yva = self.Xy_test_train_validate[ds_idx]

        classifiers = [
            (model["model_name"], model["classifier"])
            for model in self.model_results_validation if model["dataset"] == ds_idx
        ] 
       
        vote_clf = VotingClassifier( classifiers, voting=self.voting )
        
        vote_clf.fit(Xtr, ytr)
        y_hat = vote_clf.predict(Xva)
        
        if self.verbose: 
            print(f"voting_classifier : dataset {ds_idx}") 

        precision, recall, accuracy = self.calculate_scores(yva, y_hat, self.verbose)
             
        self.model_results_validation.append({
            "dataset":ds_idx, 
            "model_name":"vote_clf", 
            "classifier":vote_clf, 
            "recall":recall, 
            "precision":precision, 
            "accuracy":accuracy,
        })

    # Evaluates the scores of all models using test data instead of valiadation data
    def evaluate_models(self):
        for model in self.model_results_validation:
            Xte = self.Xy_test_train_validate[model["dataset"]][1]
            yte = self.Xy_test_train_validate[model["dataset"]][4]
            y_hat = model["classifier"].predict(Xte)
            model["precision"], model["recall"], model["accuracy"] = self.calculate_scores(yte, y_hat, self.verbose)
            self.model_results_test.append(model)

        
    def evaluate_report(self):
        bc = self.find_best_combination()
        Xy = self.Xy_test_train_validate[bc["dataset"]]
        Xte, yte = Xy[1], Xy[4]
        y_hat = bc["classifier"].predict(Xte)
        print( f"\t\t\tModel {bc["model_name"]} | Dataset {bc["dataset"]+1}", end="\n\n")
        print(classification_report(yte, y_hat))

    
    def evaluate_matrix(self):
        bc = self.find_best_combination()
        Xy = self.Xy_test_train_validate[bc["dataset"]]
        Xte, yte = Xy[1], Xy[4] 
        y_hat = bc["classifier"].predict(Xte)
        print( f"\t\t\tModel {bc["model_name"]} | Dataset {bc["dataset"]+1}", end="\n\n")
        cm = confusion_matrix(yte, y_hat)
        ConfusionMatrixDisplay(cm).plot(cmap=plt.cm.Reds)


def main():
    pass

if __name__ == "__main__":
    main()