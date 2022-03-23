from cProfile import label
import pandas as pd

#K is customizable but by default we will use K=3
K=3
###################### Training ######################
#first we will load in the training data
train_df = pd.read_csv('K_means_train.csv')
train_df = train_df.drop(["Labels","Id"],axis=1)
#pick K initial points as centroids
centroids = train_df.sample(n=K)
print(train_df)
recalculating = True
#looping:
while recalculating:
    #calculate the distance between each point and the centroids
    distances = pd.DataFrame()
    i=0
    for index, centroid in centroids.iterrows():
        dimensional_distances = ((train_df- centroid)**2)
        dimensional_distances['distance'] = (dimensional_distances.sum(axis=1))**0.5
        distances['cluster'+str(i)] = dimensional_distances['distance'] 
        i+=1
    #assign each point to the cluster it is closest to
    clusters = distances.idxmin(axis=1)
    #pick new centroids
    train_df_cluster = train_df.copy()
    train_df_cluster['label'] = clusters
    new_centroids = []
    for i in range(K):
        cluster = train_df_cluster[train_df_cluster['label']=="cluster"+str(i)]
        new_centroid = cluster.mean(axis=0)
        new_centroids.append(new_centroid.to_frame().T)
    new_centroids = pd.concat(new_centroids)
    print(new_centroids)
    #if there have been no changes since the last loop, exit the lop
    if(new_centroids.sort_values(by=['SepalLengthCm']).equals(centroids.sort_values(by=['SepalLengthCm']))):
        recalculating = False
    centroids = new_centroids
###################### Validation ######################

#load in the validation data

#classify the validation data based on our centroids

#match validation clusters to their assinged counterparts

#see what percent are mis-classified

###################### Testing ######################

#load the test data

#classify based on our centroids

#plot the clustering results using T-SNE (bonus points)