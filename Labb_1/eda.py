import numpy as np
import matplotlib.patches as mpatches


def grouping(df, group, attribute):
    # Group
    # 1) Turns your df in to a multi index series
    df_group = df.groupby([group,attribute], as_index=True)[attribute].count()
   
    # 2) Coverts it back in to a series
    df_group = df_group.to_frame("count")
    
    # 3) Flattens it, each row now consists of group, attribute, value
    df_group = df_group.reset_index()
    return df_group


def label_discreet(df, column_key_label):
    df = df.copy(deep=True)
    for key, val in column_key_label.items():
        #print(key, val)
        df[key] = df[key].map( lambda x: val[x])
    return df


def plot_cvd(df, plt, cpa):
    # Group
    # 1) Turns your df in to a multi index series
    df = df.groupby("cardio")["cardio"].count()

    # 2) Coverts it back in to a series
    df = df.to_frame("count")

    # 3) Flattens it, each row now consists of group, attribute, value
    df = df.reset_index()

    df["cardio"] = df["cardio"].map(lambda x: {0:"negative", 1:"positive",}[x])
    df.set_index("cardio", inplace=True)
    df

    fig, ax = plt.subplots(figsize=(16, 8));
    axis = ax.bar(df.index, df["count"], width=.5, color=[cpa[-3],cpa[1]], edgecolor=[cpa[-2],cpa[0]], linewidth=8)
    ax.bar_label(axis, padding=20)
    ax.set_title("CVD Postive vs Negative", fontsize=36)
    ax.set(ylim=(0, max(df["count"])*1.2), )


def plot_cholestorol(df, plt, cpa):
    cholesterol = df.groupby("cholesterol").count()
    fig = plt.figure( )
    fig = plt.pie(cholesterol["gender"], labels=["Normal", "Above Normal", "Well Above Normal"],  autopct="%1.1f%%",
                radius=0.8, wedgeprops={"edgecolor":"#ffffff",'linewidth':1, 'linestyle': '-', 'antialiased': True}, 
                textprops=dict(color="white", size=10), colors=[cpa[-3], cpa[-2], cpa[-1]  ])
    fig = plt.title("Patients  Cholestrol Levels", fontsize=12)


def plot_age(df, plt, cpa):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.hist(df["age"], bins=15, color=cpa[-3], edgecolor=cpa[-1], linewidth=1);
    ax.set_title("Age Distrubtution Among Patients")
    ax.set_ylabel("Patients")
    ax.set_xlabel("Age (Days)")
    
    # There is likely a better way to convert days in to years
    ax.set_xticks(range(df["age"].min()-1500, (df["age"].max()+2000),2150))
    ax.set_xticklabels([int(days/365.25) for days in range(df["age"].min()-1500, (df["age"].max()+2000),2150)])


def plot_smoke(df, plt, cpa):
    smoke = df.groupby("smoke").count()
    fig, ax = plt.subplots()
    ax = plt.pie(smoke["gender"], labels=["Non-Smoker", "Smoker"],  autopct="%1.1f%%", radius=0.8,
                wedgeprops={"edgecolor":"#ffffff",'linewidth':1, 'linestyle': '-', 'antialiased': True},              
                textprops=dict(color="white", size=10), colors=[cpa[-2], cpa[1]  ])
    ax = plt.title("Ratio of Smokers to Non-Smokers", fontsize=12)


def plot_gender(df, plt, cpa):
  
    group = "gender"
    attribute = "cardio"

    df = label_discreet(df, {
    "gender":{2:"female", 1:"male"},
    "cardio":{0:"positive", 1:"negative"}
    })

    # Group
    # 1) Turns your df in tu a multi index series
    df = df.groupby([group,attribute], as_index=True)[attribute].count()
    # 2) Coverts it back in to a series
    df = df.to_frame("count")
    # 3) Flattens it, each row now consists of group, attribute, value
    df = df.reset_index()
    
    width = 0.25
    x = np.arange( len(df[attribute].unique()) ) 
    
    fig, ax = plt.subplots(figsize=(16, 8))
    color=(cpa[-3], cpa[2])
    edgecolor=(cpa[-1], cpa[0])
    linewidth=4

    for n, idx in enumerate(df[attribute].unique()): 
        y = df[df[attribute] == idx]["count"]
        axis = ax.bar( x+width*n, y, width-linewidth/500,  color=color[n], edgecolor=edgecolor[n], linewidth=linewidth )
        ax.bar_label(axis)
    ax.set_xticks(x+width/len(x), df[group].unique())
    ax.set_ylim(0,max(df["count"])*1.2)
    ax.legend( labels=df[attribute].unique())
    ax = plt.title("Ratio of Males to Females CVD Results")
    plt.show()


def bmi(df):
    # Create a bmi feature with the bmi formula
    df["bmi"] = df["weight"].values / np.square( df["height"].values/100 )
    
    # Round off bmi to make it easier to read and calculate(?)
    df["bmi"] = df["bmi"].round()

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

    # Create a new discrete category
    df["bmi_category"] = switch(df["bmi"], lambda x,y: x>y, bmi_grade, "under weight")

    # Remove outliers based on bmi
    df = df[ (df["bmi"]>16) & (df["bmi"]<48) ]

    return df


def blood_preassure(df):
    # Drop outliers
    df = df[ (df["ap_hi"]>115*0.7) & (df["ap_hi"]<185*1.1) ]
    df = df[ (df["ap_lo"]>75-20) & (df["ap_lo"]<110+20) ]

    # 1) In order to combine ap_hi and ap_lo in to one value so that they can 
    #    be evalueate togehter one of them has to be rescaled so both 
    #    Systolic and Diastolic blood preassure holds equal weight
    #    (100/120)*80  ~ 62.5
    #    (100/180)*120 ~ 62.5
    scale = 1.625
   
    # 2) Scale ap_lo and combine with ap_hi in order to get one value to evaluate
    exp_combine = lambda row: row["ap_hi"] + row["ap_lo"]*scale
    values = df.apply(exp_combine, axis=1)

    # 3) Creates a dictionary where the keys are the combined ap_hi + ap_lo * scale 
    #    and the values are the grades
    los = (120, 90, 80, 80, 0)
    his = (180, 140, 130, 120, 0)
    grades = ("Hypertension Crisis", "Hypertension Stage 2", "Hypertension Stage 1", "Elevated", "Healthy")
    blood_preasure_grade = { (lo+hi*scale):grade for lo, hi, grade in zip(los, his, grades) }

    # 4) Using list comprehension each value is checked agains each key, 
    #    if the value is greater then the key the key is returned 
    #    we then grab the highest value key and use it to return
    #    the coresponding value from the dictionary.
    results = [ blood_preasure_grade[max( key for key in blood_preasure_grade.keys() if val > key)] for val in values ]

    df["bp_category"] = results
    
    return df


# Heatmap
def plot_heatmap(df, plt, drop):
    df_heatmap = df.drop( columns=drop)
    # Connverts bp_category in to numbers
    cat = { "Hypertension Crisis":4, "Hypertension Stage 2":3, "Hypertension Stage 1":2, "Elevated":1, "Healthy":0}
    df_heatmap["bp_category"] = df_heatmap["bp_category"].map( lambda x: cat[x] ) 

    cor = df_heatmap.corr()
    col = list(df_heatmap.columns)

    fig, ax = plt.subplots( figsize=(16, 8))
    im = ax.imshow(df_heatmap.corr(), cmap="Reds")
    ax.set_xticks(range(len(col)), labels=col, rotation=45, ha="right", rotation_mode="anchor" );
    ax.set_yticks(range(len(col)), labels=col);

    # Adds the numbers to the squares
    ncorr = np.round(cor.to_numpy()*100)/100
    idx = df_heatmap.index
    for i in range(ncorr.shape[0]):
        for j in range(ncorr.shape[1]):
            text = ax.text(j,i, ncorr[i,j], ha="center", va="center", color="black", fontsize=6)


# Visualizing CVD with Subplots 
def plot_subpots(df, plt, cpa):
   
    df_group = label_discreet(df, {
        "cardio":{0:"positive", 1:"negative"},
        "alco":{0:"non-drinker", 1:"drinker"},
        "smoke":{0:"non-smoker", 1:"smoker"},
        "cholesterol":{1:"normal", 2:"above normal", 3:"well above normal"},
        "active":{0:"not physically active", 1:"physically active"},
        "gluc":{1:"normal", 2:"above normal", 3:"well above normal"},
        
    })

    x_labels=[
        ["normal", "obese (class 1)", "over weight", "obese (class 2)", "under weight", "obese (class 3)"],
        ["normal", "well above normal", "above normal"],
        ["normal", "above normal", "well above normal"],
        ["non-smoker", "smoker"],
        ["non-drinker", "drinker"],
        ["not physically active", "physically active"]
    ]

    features=["bmi_category", "cholesterol", "gluc", "smoke", "alco", "active"]
    titles=["BMI Categories", "Cholesterol Categories", "Gluconse Categories", "Smoker or Non-Smoker", "Drinker or Non-Drinker", "Physical Activity",]

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(32, 16))
    color=(cpa[-3], cpa[2])
    edgecolor=(cpa[-1], cpa[0])
    linewidth=4
    width = 0.10
    ax=ax.flatten()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    

    for i, axi in enumerate(ax):
        df_group = grouping(df, "cardio", features[i])

        x = np.arange( len(df["cardio"].unique()) ) 
        xx = np.arange( len(df[features[i]].unique()) ) 

        for j, idx in enumerate(df[features[i]].unique()): 
            y = df_group[df_group[features[i]] == idx]["count"]
            axis =axi.bar( (x+j)+np.array([0,-0.89]), y, width-linewidth/500,  color=color, edgecolor=edgecolor, linewidth=linewidth )
            axi.bar_label(axis)

        axi.set_xticks(xx+width/2, x_labels[i], rotation=15)
        axi.set_ylim(0,max(df_group["count"])*1.2)
        blue_patch = mpatches.Patch(color=color[0], label="Negative")
        red_patch = mpatches.Patch(color=color[1], label="Positive")

        axi.legend( handles=[blue_patch, red_patch])
        axi.set_title(titles[i], fontsize=24)

    plt.show()


def plot_height_weight(df, plt, cpa):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
    ax1.hist(df["height"], bins=60, color=cpa[-3], edgecolor=cpa[-1], linewidth=1)

    ax1.set_ylabel("Number of Patients")
    ax1.set_title("Height (cm)", fontsize=18)
    ax2.hist(df["weight"], bins=60, color=cpa[2], edgecolor=cpa[0], linewidth=1)

    ax2.set_ylabel("Number of Patients")
    ax2.set_xlabel("Weight (kg)")
    plt.tight_layout()
    plt.show()


def main():
    pass

if __name__ == "__main__":
    main()