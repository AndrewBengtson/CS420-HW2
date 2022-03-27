import pandas as pd


K=10

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
print(findKNN(training_df,valid_df,K,"Classify_Labels"))
###################### Testing ######################
#load in the testing data set
testing_df = pd.read_csv("KNN_test.csv")
#call a KNN helper function
print(findKNN(training_df,testing_df,K,"Labels"))
