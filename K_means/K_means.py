from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def show_output(df,non_numberic,save_name):
    m = TSNE()
    features = m.fit_transform(non_numberic)
    df['x'] = features[:,0]
    df['y'] = features[:,1]

    sns.scatterplot(x='x',y='y',data=df,hue=df['label'],style=df['data_type'],size=df['data_type'])
    plt.savefig(save_name)


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
    #if there have been no changes since the last loop, exit the lop
    if(new_centroids.sort_values(by=['SepalLengthCm']).equals(centroids.sort_values(by=['SepalLengthCm']))):
        recalculating = False
    centroids = new_centroids
###################### Validation ######################

#load in the validation data
valid_df = pd.read_csv("K_means_valid.csv")
#classify the validation data based on our centroids
#calculate the distance between each point and the centroids
distances = pd.DataFrame()
i=0
for index, centroid in centroids.iterrows():
    dimensional_distances = ((valid_df- centroid)**2)
    dimensional_distances['distance'] = (dimensional_distances.sum(axis=1))**0.5
    distances['cluster'+str(i)] = dimensional_distances['distance'] 
    i+=1
#assign each point to the cluster it is closest to
clusters = distances.idxmin(axis=1)
valid_df_cluster = valid_df.copy()
valid_df_cluster['label'] = clusters

###################### Testing ######################

#load the test data
test_df = pd.read_csv('K_means_test.csv')
#classify based on our centroids
#calculate the distance between each point and the centroids
distances = pd.DataFrame()
i=0
for index, centroid in centroids.iterrows():
    dimensional_distances = ((test_df- centroid)**2)
    dimensional_distances['distance'] = (dimensional_distances.sum(axis=1))**0.5
    distances['cluster'+str(i)] = dimensional_distances['distance'] 
    i+=1
#assign each point to the cluster it is closest to
clusters = distances.idxmin(axis=1)
test_df_cluster = test_df.copy()
test_df_cluster['label'] = clusters
#print the estimations
print(test_df_cluster.drop(['labels'],axis=1).to_string())
#plot the clustering results using T-SNE (bonus points)
#add a column which tells us which dataset each comes from
print(train_df_cluster.count()['label'])
train_df_cluster["data_type"] = ["train"] * train_df_cluster.count()['label']
test_df_cluster["data_type"] = ["test"] *test_df_cluster.count()['label']
valid_df_cluster["data_type"] = ["valid"] *valid_df_cluster.count()['label']
print(pd.concat([train_df_cluster,test_df_cluster,valid_df_cluster]))
show_output(pd.concat([train_df_cluster,test_df_cluster,valid_df_cluster]),pd.concat([train_df_cluster,test_df_cluster,valid_df_cluster]).drop(['label','labels','Id',"Labels","data_type"],axis=1),"Test_out.png")