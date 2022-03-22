import pandas as pd

#K is customizable but by default we will use K=3
K=3
###################### Training ######################
#first we will load in the training data
train_df = pd.read_csv('K_means_train.csv')
#pick K initial points as centroids

#looping:

    #calculate the distance between each point and the centroids

    #assign each point to the cluster it is closest to

    #pick new centroids
    
    #if there have been no changes since the last loop, exit the lop

###################### Validation ######################

#load in the validation data

#classify the validation data based on our centroids

#match validation clusters to their assinged counterparts

#see what percent are mis-classified

###################### Testing ######################

#load the test data

#classify based on our centroids

#plot the clustering results using T-SNE (bonus points)