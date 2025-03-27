import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

path = "data/"
df = pd.read_csv(path+"cardio_train.csv", sep=";")
df["year"] = df["age"].map(lambda days: int(days/365.25) )
df.drop(columns="id", inplace=True)
df



custome_style={
    "axes.titlecolor":"ffffff",
    "axes.labelcolor":"ffffff",
    "axes.edgecolor": "ffffff", # Spine
    "axes.facecolor": "1f1f1f", 

    "axes.titlesize":12,

    "font.size":20,
    "text.color":"ffffff",

    "figure.facecolor":"1f1f1f", 
    "figure.dpi":150,
    "ytick.color": "ffffff",
    "ytick.labelsize": 15,
    "xtick.color": "ffffff",
    "xtick.labelsize": 15,
    

    "axes.axisbelow": True, # Makes sure the grid is behind the plots
    "axes.grid": True,
    'grid.alpha': .2,
    'grid.color': 'ffffff',
    'grid.linestyle': '-',
    'grid.linewidth': 1,

    'legend.labelcolor': 'ffffff', # Font color
    'legend.edgecolor': 'ffffff', # border
    'legend.fontsize': 16,


    "axes.edgecolor": "ffffff",

    'boxplot.capprops.color': 'ffffff', # The top of the T line
    'boxplot.whiskerprops.color': 'ffffff',# The middle of the T line
    'boxplot.boxprops.color': 'ffffff', # The box line
    'boxplot.medianprops.color': 'ffffff',
    'boxplot.medianprops.linestyle': '-',
    'boxplot.medianprops.linewidth': 1.0,

    'boxplot.flierprops.marker': 'o',   
    'boxplot.flierprops.markeredgecolor': '1f1f1f',
    'boxplot.flierprops.markeredgewidth': .4,
    'boxplot.flierprops.markerfacecolor': 'ffffff', # set for each boxplot
    'boxplot.flierprops.markersize': 4.0,
}

for key, val in custome_style.items(): plt.rcParams[key]=val

cpa="#8C0000,#A80000,#C60013,#E32227,#0091C5,#0079AC,#006194".split(",")
wpr={"edgecolor":"#ffffff",'linewidth':1, 'linestyle': '-', 'antialiased': True}


def plot_cvd_pos_neg():
    # Group
    # 1) Turns your df in to a multi index series
    dfg = df.groupby("cardio")["cardio"].count()

    # 2) Coverts it back in to a series
    dfg = dfg.to_frame("count")

    # 3) Flattens it, each row now consists of group, attribute, value
    dfg = dfg.reset_index()

    dfg["cardio"] = dfg["cardio"].map(lambda x: {0:"negative", 1:"positive",}[x])
    dfg.set_index("cardio", inplace=True)
    dfg

    fig, ax = plt.subplots(figsize=(16, 8));
    axis = ax.bar(dfg.index, dfg["count"], width=.5, color=[cpa[-3],cpa[1]], edgecolor=[cpa[-2],cpa[0]], linewidth=8)
    ax.bar_label(axis, padding=20)
    ax.set_title("CVD Postive vs Negative", fontsize=24)
    ax.set(ylim=(0, max(dfg["count"])*1.2), )


def plot_cholestorol():
    cholesterol = df.groupby("cholesterol").count()
    fig = plt.figure( )
    fig = plt.pie(cholesterol["gender"], labels=["Normal", "Above Normal", "Well Above Normal"],  autopct="%1.1f%%",
                radius=0.8, wedgeprops=wpr, 
                textprops=dict(color="white", size=10), colors=[cpa[-3], cpa[-2], cpa[-1]  ])
    fig = plt.title("Patiens Cholestrol Values", fontsize=12)


def plot_age():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.hist(df["age"], bins=15, color=cpa[-3], edgecolor=cpa[-1], linewidth=1);
    ax.set_title("Age Distrubtution Among Patients")
    ax.set_ylabel("Patients")
    ax.set_xlabel("Age (Days)")


def plot_smoke():
    smoke = df.groupby("smoke").count()
    fig, ax = plt.subplots()
    ax = plt.pie(smoke["gender"], labels=["Non-Smoker", "Smoker"],  autopct="%1.1f%%", radius=0.8,
                wedgeprops={"edgecolor":"#ffffff",'linewidth':1, 'linestyle': '-', 'antialiased': True},              
                textprops=dict(color="white", size=10), colors=[cpa[-1], cpa[0]  ])
    ax = plt.title("Ratio of Smokers to Non-Smokers")


def plot_height_weight():
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
    ax1.hist(df["height"], bins=60, color=cpa[-3], edgecolor=cpa[-1], linewidth=1)
    #ax1.set_title("Height Distribution")
    ax1.set_ylabel("Number of Patients")
    ax1.set_title("Height (cm)", fontsize=18)
    ax2.hist(df["weight"], bins=60, color=cpa[2], edgecolor=cpa[0], linewidth=1)
    #ax2.set_title("Weight Distribution")
    ax2.set_ylabel("Number of Patients")
    ax2.set_xlabel("Weight (kg)")
    plt.tight_layout()
    plt.show()


def label_discreet(df, column_key_label):
    df = df.copy(deep=True)
    for key, val in column_key_label.items():
        #print(key, val)
        df[key] = df[key].map( lambda x: val[x])
    return df

dfg = label_discreet(df, {
    "gender":{2:"female", 1:"male"},
    "cardio":{0:"positive", 1:"negative"}
})



def group_and_plot(df, group, attribute):
    df = df.copy( deep=True )

    # Group
    # 1) Turns your df in tu a multi index series
    df = df.groupby([group,attribute], as_index=True)[attribute].count()
    # 2) Coverts it back in to a series
    df = df.to_frame("count")
    # 3) Flattens it, each row now consists of group, attribute, value
    df = df.reset_index()
    
    #print(df)
    # Plot 
    # 1) 
    width = 0.25
    x = np.arange( len(df[attribute].unique()) ) 
    fig, ax = plt.subplots(figsize=(16, 8))
    color=(cpa[-3], cpa[2])
    edgecolor=(cpa[-1], cpa[0])
    linewidth=4

    for n, idx in enumerate(df[attribute].unique()): 
        y = df[df[attribute] == idx]["count"]
        #print(f"{y}")
        axis = ax.bar( x+width*n, y, width-linewidth/500,  color=color[n], edgecolor=edgecolor[n], linewidth=linewidth )
        ax.bar_label(axis)
    ax.set_xticks(x+width/len(x), df[group].unique())
    ax.set_ylim(0,max(df["count"])*1.2)
    ax.legend( labels=df[attribute].unique())
    plt.show()


# BMI

df["bmi"] = df["weight"].values / np.square( df["height"].values/100 )
def drop_outliers(df, column):
    return df[ (df[column]>16) & (df[column]<48) ]
df = drop_outliers(df, "bmi")



# Switch 
# 1) Loops through a dict 
# 2) uses its keys in an lambda if test 
# 3) if the tests succeeds the value from the key value pair is picked
# 4) if it fails the result is set to the default value
def switch(values, exp, di, default):
    results = list()
    for val in values:
        for key in di.keys():
            result = di[key] if exp(val,key) else default 
            if result != default : break
        results.append(result) 
    return results

bmi_grade = {
    40:"obese (class 3)",
    35:"obese (class 2)",
    30:"obese (class 1)",
    25:"over weight",
    18.5:"normal range",
}

df["bmi_category"] = switch(df["bmi"], lambda x,y: x>y, bmi_grade, "under weight")






# Blood Preassure

# Drop outliers
df = df[ (df["ap_hi"]>115*0.7) & (df["ap_hi"]<185*1.1) ]
df = df[ (df["ap_lo"]>75-20) & (df["ap_lo"]<110+20) ]

# Switch 
# 1) Loops through a dict 
# 2) uses its keys in an lambda if test 
# 3) if the tests succeeds the value from the key value pair is picked
# 4) if it fails the result is set to the default value
def switch(df, values, exp, di, default):
    results = list()
    for row in df: 
        for key in di.keys():
            result = di[key] if exp(val,key) else default 
            if result != default : break
        results.append(result) 
    return results

scale = 1.65


# SYSTOLIC|DIASTOLIC|BLOOD PREASSURE CATEGORY
blood_preasure_grade = {
    (180, 120):"Hypertension Crisis",
    (140, 90):"Hypertension Stage 2",
    (130, 80):"Hypertension Stage 1",
    (120, 80):"Elevated",
}

#switch( df=df, values=("ap_hi", "ap_lo"), exp=,  di=blood_preasure_grade,  default="Healthy")


# (100/120)*80  ~ 62.5
# (100/180)*120 ~ 62.5

# 1) In order to combine ap_hi and ap_lo in to one value so that they can 
#    be evalueate togehter one of them has to be rescaled so they both 
#    hold equal weight
# (100/120)*80  ~ 62.5
# (100/180)*120 ~ 62.5
# scale = 1.625
scale = 1.65


# 2) Scale ap_lo and combine with ap_hi in order to get one value to evaluate
exp_combine = lambda row: row["ap_hi"] + row["ap_lo"]*scale
values = df.apply(exp_combine, axis=1)

# 3) Creates a dictionary where the keys are the combined ap_hi + ap_lo*scale 
#    and the values are the grades
los = (120, 90, 80, 80, 0)
his = (180, 140, 130, 120, 0)
grades = ("Hypertension Crisis", "Hypertension Stage 2", "Hypertension Stage 1", "Elevated", "Healthy")
blood_preasure_grade = { (lo+hi*scale):grade for lo, hi, grade in zip(los, his, grades) }

# 4) Using list comprehension we check each value agains each key, 
#    if the value is greater then the key the key is returned 
#    we then grab the highest value key and use it to return
#    the coresponding value from the dictionary.
results = [ blood_preasure_grade[max( key for key in blood_preasure_grade.keys() if val > key)] for val in values ]

df["bp_category"] = results





# Visualizing CVD with Subplots 
def grouping(df, group, attribute):
    # Group
    # 1) Turns your df in to a multi index series
    df_group = df.groupby([group,attribute], as_index=True)[attribute].count()
    # 2) Coverts it back in to a series
    df_group = df_group.to_frame("count")
    # 3) Flattens it, each row now consists of group, attribute, value
    df_group = df_group.reset_index()
    return df_group

def plot(df, group, attributes, x_labels):
    
    width = 0.10
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(32, 16))
    color=(cpa[-3], cpa[2])
    edgecolor=(cpa[-1], cpa[0])
    linewidth=4
    ax=ax.flatten()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    titles=["BMI Categories", "Blood Preasure Categories", "Cholesterol Categories", "Gluconse Categories", "Smoker or Non-Smoker", "Drinker or Non-Drinker"]

    for i, axi in enumerate(ax):
        df_group = grouping(df, group, attributes[i])
        #print(df_group)
        #print(df_group[df_group.columns[1]])
        #print(df_group)
        x = np.arange( len(df[group].unique()) ) 
        xx = np.arange( len(df[attributes[i]].unique()) ) 
        for j, idx in enumerate(df[attributes[i]].unique()): 
      
            y = df_group[df_group[attributes[i]] == idx]["count"]
            #print(y)
            #print((x+j)+np.array([1,.1]))

            axis =axi.bar( (x+j)+np.array([0,-0.89]), y, width-linewidth/500,  color=color, edgecolor=edgecolor, linewidth=linewidth )

            axi.bar_label(axis)
        axi.set_xticks(xx+width/2, x_labels[i], rotation=15)
        axi.set_ylim(0,max(df_group["count"])*1.2)
        blue_patch = mpatches.Patch(color=color[0], label="Negative")
        red_patch = mpatches.Patch(color=color[1], label="Positive")
        #axi.legend( labels=df_group[group].unique())
        axi.legend( handles=[blue_patch, red_patch])
        axi.set_title(titles[i], fontsize=24)
    plt.show()


df_group = label_discreet(df, {
    "cardio":{0:"positive", 1:"negative"},
    "alco":{0:"non-drinker", 1:"drinker"},
    "smoke":{0:"non-smoker", 1:"smoker"},
    "cholesterol":{1:"normal", 2:"above normal", 3:"well above normal"},
    "gluc":{1:"normal", 2:"above normal", 3:"well above normal"},
})

x_labels=[
    ["normal", "obese (class 1)", "over weight", "obese (class 2)", "under weight", "obese (class 3)"],
    ["healthy", "elevated", "hyper stage 1", "hyper stage 2"],
    ["normal", "well above normal", "above normal"],
    ["normal", "above normal", "well above normal"],
    ["non-smoker", "smoker"],
    ["non-drinker", "drinker"],
]

features=["bmi_category", "bp_category", "cholesterol", "gluc", "smoke", "alco"]




# Heatmap
def plot_heatmap():
    df_heatmap = df.drop( columns=[ "year", "bmi_category"] )
    # Should probably use dummy
    cat = { "Hypertension Crisis":4, "Hypertension Stage 2":3, "Hypertension Stage 1":2, "Elevated":1, "Healthy":0}
    df_heatmap["bp_category"] = df_heatmap["bp_category"].map( lambda x: cat[x] ) 


    df_heatmap.corr()
    cor = df_heatmap.corr()
    fig, ax = plt.subplots( figsize=(16, 8))
    im = ax.imshow(df_heatmap.corr(), cmap="Reds")
    col=list(df_heatmap.columns)
    ax.set_xticks(range(len(col)), labels=col, rotation=45, ha="right", rotation_mode="anchor" );
    ax.set_yticks(range(len(col)), labels=col);

    ncorr = np.round(cor.to_numpy()*100)/100
    #ncorr = cor.to_numpy()
    # text = ax.text(ncorr[:,0],ncorr[0,:], ncorr, ha="center", va="center", color="w")
    idx = df_heatmap.index
    for i in range(ncorr.shape[0]):
        for j in range(ncorr.shape[1]):
            text = ax.text(j,i, ncorr[i,j], ha="center", va="center", color="black", fontsize=6)
    ncorr


#----------------------------------

# Create Two Datasets
df_a = df.drop(columns=["ap_hi", "ap_lo", "height", "weight", "bmi", "year"])
df_a = pd.get_dummies(df_a, drop_first=True)
df_a.head()


df_b = df.drop(columns=["bmi", "bmi_category", "bp_category", "height", "weight", "year"])
df_b = label_discreet(df_b, {"gender":{2:"female", 1:"male"}})
df_b = pd.get_dummies(df_b, drop_first=True)
df_b.head()


# Creates a list continga multiple lists of X & y with labels
def XYL_formating(df, X_labels, y_labels,  size ):
    XYL = [ 
        {"X":df[features].iloc[:size].values, "y":df[target].iloc[:size].values, "X_labels":features, "y_label":target} 
        for features, target in zip(X_labels, y_labels) ]
    return XYL

#size = 200
size = 70000

X_labels = [
    ["height", "weight", "age"], 
    ["ap_hi", "ap_lo", "gender"]
]
y_labels = [ "cardio", "cardio"]
XYL = XYL_formating(df, X_labels, y_labels, size)





from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix


class ModelValidator:
    def __init__(self, XYL):
        self.XYL = XYL  # Contains the X, y and labels for each data set
        self.Xy_test_train_validate = list()
        #self.scaler = StandardScaler()
        self.verbose = False 
        self.score = "accuracy"
        self.scaler = MinMaxScaler()
        self.normalizer = Normalizer()
        self.model_results = list()
        self.classifiers = list()
        

        

    def split(self, X, y):
        # tr = Train 
        # va = Validation (used during tuning of hyper parameters)
        # te = Test (used ofr the final test) 
        Xtr, Xte, ytr, yte  =  train_test_split(X, y, test_size=.3)
        Xte, Xva, yte, yva  =  train_test_split(Xte, yte, test_size=.5)
        self.Xy_test_train_validate.append([Xtr, Xte, Xva, ytr, yte, yva]) 
    



    # These exists in the sklearn library, 
    # but I wanted to make it a little more clear
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


    # Use max to pick the model data-set combo with highest score 
    def find_best_combination(self):
        return max(self.model_results, key=lambda di: di[self.score])


    def initiat_model_tuning(self):
        for ds_idx, data_set in enumerate(XYL):
            self.classifiers.append(list())
            self.split(data_set["X"], data_set["y"])
            for model in ["gradient_boosting", "decision_tree", "knn", "logistic_regression", "multinomial_nb"]:  
            #for model in ["gradient_boosting", "decision_tree"]: 
                self.execute_model_tuning(model, ds_idx)
            self.voting_classifier(ds_idx)


    def execute_model_tuning(self, model, ds_idx):
        Xtr, Xte, Xva, ytr, yte, yva = self.Xy_test_train_validate[ds_idx]

        models={
            "decision_tree":DecisionTreeClassifier,
            "gradient_boosting":GradientBoostingClassifier,
            "knn":KNeighborsClassifier,
            "logistic_regression":LogisticRegression, 
            "multinomial_nb":MultinomialNB, 
        }

        # The hyper parameters I got from an LLM 
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

        pipe = Pipeline([
            ("scaler", self.scaler),
            ("normalizeer", self.normalizer), 
            (model, models[model]())
        ]) 


        classifier = GridSearchCV(
            estimator=pipe, 
            param_grid=param_grid[model], 
            cv=5, #Alsoe try 10 
            scoring=self.score,
            n_jobs=-1,
            verbose=self.verbose
        )
        
        if self.verbose: 
            print(f"{model} : dataset {ds_idx}") 
        classifier.fit(Xtr, ytr)           
        best_version = classifier.best_estimator_
        y_hat = best_version.predict(Xva)

        precision, recall, accuracy = self.calculate_scores(yva, y_hat, self.verbose)

        # Stores all the information about the best model in a list(dic())
        # this allows us to pick the best model/ds pair with the max() function
        # and we store the entire method so that we can 
        # impliment the best model dynamically using only the information in this 
        # model_result + the test data.
        
        self.model_results.append({
            "data_set":ds_idx, 
            "model_name":model,  
            "classifier":best_version,
            "recall":recall, 
            "precision":precision, 
            "accuracy":accuracy,
            "Xte":Xte, 
            "yte":yte,

        })
        
        # This information already exists in model_results
        # but its stored in a way that makes it easier 
        # to implement for the voting classifier 
        self.classifiers[ds_idx].append( (model, best_version) )

    def voting_classifier(self, ds_idx):
        
        Xtr, Xte, Xva, ytr, yte, yva = self.Xy_test_train_validate[ds_idx]
       
        vote_clf = VotingClassifier( self.classifiers[ds_idx], voting="hard" )
        vote_clf.fit(Xtr, ytr)
        y_hat = vote_clf.predict(Xva)
        
        if self.verbose: 
            print(f"voting_classifier : dataset {ds_idx}") 
        precision, recall, accuracy = self.calculate_scores(yva, y_hat, self.verbose)
             
        self.model_results.append({
            "data_set":ds_idx, 
            "model_name":"vote_clf", 
            "classifier":vote_clf, 
            "recall":recall, 
            "precision":precision, 
            "accuracy":accuracy,
            "Xte":Xte, 
            "yte":yte,
        })
    

    def evaluate_report(self):
        bc = self.find_best_combination()
        Xte, yte = bc["Xte"], bc["yte"]  
        y_hat = bc["classifier"].predict(Xte)
        print( f"\t\t\tModel {bc["model_name"]} | Dataset {bc["data_set"]}", end="\n\n")
        print(classification_report(yte, y_hat))
    
    def evaluate_matrix(self):
        bc = self.find_best_combination()
        Xte, yte = bc["Xte"], bc["yte"]  
        y_hat = bc["classifier"].predict(Xte)
        print( f"\t\t\tModel {bc["model_name"]} | Dataset {bc["data_set"]}", end="\n\n")
        cm = confusion_matrix(yte, y_hat)
        ConfusionMatrixDisplay(cm).plot( ) # .plot(cmap=plt.cm.Blues)







def main():
    pass

if __name__ == "__main__":
    main()