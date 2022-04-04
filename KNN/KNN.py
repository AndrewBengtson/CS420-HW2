from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import pickle
import time

start = time.time()
K=20
print("K = ",K)
#Helper function that produces graphs using TSNE
def show_output(df,non_numberic,save_name):
    #If we already have the features file, use it
    if(os.path.exists("features.pkl")):
        features = pickle.load(open('features.pkl','rb'))
    #if features hasn't been trained yet, make a new one
    else:
        m = TSNE()
        features = m.fit_transform(non_numberic)
        pickle.dump(features,open('features.pkl','ab'))
    df['x'] = features[:,0]
    df['y'] = features[:,1]

    sns.scatterplot(x='x',y='y',data=df,hue=df['Labels'],style=df['data_type'],size=df['data_type'])
    plt.savefig(save_name)


#This helper function is going to find the K nearest neighbors
def findKNN(training_df,classify_df,K,output_name):
    labels = []
    #for each row in the classification df
    for index, row in classify_df.drop(['Id',"Labels"],axis=1).iterrows():
        #estimate it's distance to each point in the training_df
        dimensional_distances = ((training_df.drop(['Id',"Labels"],axis=1)- row)**2)
        dimensional_distances['distance'] = (dimensional_distances.sum(axis=1))**0.5
        dimensional_distances['Labels'] = training_df["Labels"]
        #select it's K closest neighbors and select the label shared by most of the nearest neighbors
        labels.append(dimensional_distances.nsmallest(K,'distance')["Labels"].mode()[0])
    classify_df[output_name]=labels
    return classify_df
    

###################### Training ######################
#load in the training data set
training_df = pd.read_csv("KNN_train.csv")
###################### Validating ######################
#load in the validating data set
valid_df = pd.read_csv("KNN_valid.csv")
#call a KNN helper function
valid_df = findKNN(training_df,valid_df,K,"Classify_Labels")
print("validation data")
print(valid_df)
###################### Testing ######################
#load in the testing data set
testing_df = pd.read_csv("KNN_test.csv")
#call a KNN helper function
testing_df = findKNN(training_df,testing_df,K,"Labels")
print("testing data")
print(testing_df)
#print graphs with TSNE
#add a column which tells us which dataset each comes from
training_df["data_type"] = ["train"] * training_df.count()['Labels']
testing_df["data_type"] = ["test"] *testing_df.count()['Labels']
valid_df["data_type"] = ["valid"] *valid_df.count()['Labels']
show_output(pd.concat([training_df,testing_df,valid_df]),pd.concat([training_df,testing_df,valid_df]).drop(['Labels',"Classify_Labels",'Id',"data_type"],axis=1),"K="+str(K)+".png")
end = time.time()
print("runtime was "+str(end-start)+" seconds")