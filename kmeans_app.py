# Import dependencies
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import seaborn as sns
import pandas as pd

# 1.0 Defining algorithm from scratch 
def centroid_initialization(X,K):
    random_index=np.random.permutation(X.shape[0],)
    centroids=X[random_index[0:K],:]
    return centroids

def closestCentroids(centroids,X):
    idx=np.zeros((X.shape[0],1),dtype=int) #Int type for indexing later
    for i in range(0,X.shape[0]):
        error=np.zeros(centroids.shape[0])
        for j in range(0,centroids.shape[0]):
            error[j]=np.sqrt(np.sum((X[i,:]-centroids[j,:])**2))
        index=np.argmin(error)
        idx[i]=index
    return idx

def computeCentroids(X,idx,K):
    centroids=np.zeros((K,X.shape[1]))
    for i in range(0,K):
        sum=np.zeros((1,X.shape[1]))
        count=0
        for j in range(0,X.shape[0]):
            if i==idx[j]:
                sum+=X[j,:]
                count+=1
        centroids[i,:]=np.divide(sum,count);
    return centroids 

def runKMeans(X,K,max_iter):
    centroids=centroid_initialization(X,K)
    for iter in range(0,max_iter):
        dist=0
        print(f'K Means iteration {iter}/{max_iter}')
        idx=closestCentroids(centroids,X)
        centroids=computeCentroids(X,idx,K)
    return centroids,idx

def runKMeans_(X,K,max_iter,initial_centroid):
    centroids=initial_centroid
    for iter in range(0,max_iter):
        dist=0
        print(f'K Means iteration {iter}/{max_iter}')
        idx=closestCentroids(centroids,X)
        centroids=computeCentroids(X,idx,K)
    return centroids,idx

st.header('What is the K Means cluster algorithm?')
st.write('''
**Unsupervised learning algorithm:**

Unlike regression or classification which requires labelled data to predict and classify, K means 
learns the similarity of datapoints based on the data itself and does not require any labelling. 
There is not concrete 'answers' to learn from or benchmark against. 

**Cluster algorithm:**

Groups data based on their similarity. The goal of the algorithm is to minimize the error function. 
The objective and output of k means can be ambiguous compared to classic regression and classification problems. 
Regression and classification algorithms output the prediction value for things like house prices, weather, if someone has x disease etc. while clustering 
outputs the group each datapoints belong to. It is up to us to give meaning to the grouping.

**Error function:**

Total distance between each datapoint and their respective centroid. ''')
st.latex(r'''Error= \sqrt{\sum_{i=1}^{n}(q_i-p_i)^2 } ''')

st.write('''
**Parameters:** 

1. Number of groups, K: 

The number of groups we want KMeans to seperate the data into based on use case and application. 
Take the customer segmentation example, if the marketing team wants to create different emails for different segments, 
we want K to be large enough for the emails to be as relevant and personalized while being small enough to be managed by the marketing team. 

After deciding on a range for K, we can visualize how the error vary with K and choose K with a low error. Elbow method: Plotting error against K, 
and finding the best value of K such that error is significantly reduced before it plateaus. 

2. Initial centroids:

Different starting centroids will converge differently. Some might need more iterations to reach the optimized solution and 
some might get stuck in a solution that is not the best solution. (try 0,0  0,0.5, 0,1 initial centroid below)
To prevent bad clustering results from bad starting centroids,
the best practice is to run the algorithm a few times with different random centroids and choose the ones with
the lowest error. 

3. Number of iteration:

The number of times we want the algorithm to repeat the optimization steps. We should optimize this number so that it is 
small enough to not waste unneccesary time training the model but large enough for the model to reach a optimal solution. 
Again, checking the error value for every iteration might help for a large problem. We can stop iterating when error value 
starts to plateau. 
 ''')

st.header('How does K Means perceive similarity?')
st.write('''The smaller the distance between the data points, the more similar they are. The easiest way to visualize this is to imagine a bunch of points on a 2D 
graph and KMeans groups the data based on how close the datapoints are to each other. As the dimension of the data increases, eg. in customer segmentation 
use case where each customer is described by a large number of features like purchase history, sex, age, location, activity etc. it will start to get difficult 
to visualize the similarity of the data by eye but it is still possible to segment the customers with domain knowledge. However, this way of grouping does not scale
well to large data. K Means mimick this type of grouping, the distance metric
can scale up to high demensions with the distance formula: ''')
st.latex(r'''Distance= \sqrt{\smash[b]{(x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2 + ...}} ''')


st.header('K means step by step')
st.write('''
1. Scale the data so that all features has the same range. 0-1 or -1 to 1
2. Create k number of datapoints (known as centroids) with the same dimensions as our data. ''')
st.code(''' def centroid_initialization(X,K):
    random_index=np.random.permutation(X.shape[0],)
    centroids=X[random_index[0:K],:]
    return centroids''')

st.write('''3. Assigning data points to their nearest centroids.''')
st.code(''' def closestCentroids(centroids,X):
    idx=np.zeros((X.shape[0],1),dtype=int) #Int type for indexing later
    for i in range(0,X.shape[0]):
        error=np.zeros(centroids.shape[0])
        for j in range(0,centroids.shape[0]):
            error[j]=np.sqrt(np.sum((X[i,:]-centroids[j,:])**2))
        index=np.argmin(error)
        idx[i]=index
    return idx ''')
st.write('4. Reposition the centroids as the mean position of all of their datapoints')
st.code(''' def computeCentroids(X,idx,K):
    centroids=np.zeros((K,X.shape[1]))
    for i in range(0,K):
        sum=np.zeros((1,X.shape[1]))
        count=0
        for j in range(0,X.shape[0]):
            if i==idx[j]:
                sum+=X[j,:]
                count+=1
        centroids[i,:]=np.divide(sum,count);
    return centroids ''')
st.write('5. Repeat the steps for x number of iterations')
st.code(''' def runKMeans(X,K,max_iter):
    centroids=centroid_initialization(X,K)
    for iter in range(0,max_iter):
        dist=0
        print(f'K Means iteration {iter}/{max_iter}')
        idx=closestCentroids(centroids,X)
        centroids=computeCentroids(X,idx,K)
    return centroids,idx''' )


st.header('K means cluster in action!')
data=scipy.io.loadmat('data.mat');
sample_x=data['X'];

# User choose initial centroids
st.subheader('Tune the parameters! ')
st.write('Choose a set of starting centroid, only select 1 please! ')
initial_centroids=np.array([[3,3],[2,1],[5,6]]);
checkbox1=st.checkbox('[0,0 ]     [0,0]     [0,0]')
checkbox2=st.checkbox('[3,3]      [2,1]     [5,6]')
checkbox3=st.checkbox('[3,3]     [3,0]     [3,6]')
checkbox4=st.checkbox('[0,3]    [3,3]     [8,3]')
checkbox5=st.checkbox('[0,0]     [0,0.5]     [0,1]')
if checkbox1:
    initial_centroids=np.array([[0,0],[0,0],[0,0]])
if checkbox2:
    initial_centroids=np.array([[3,3],[2,1],[5,6]])
if checkbox3:
    initial_centroids=np.array([[3,3],[3,0],[3,6]])
if checkbox4:
    initial_centroids=np.array([[0,3],[3,3],[8,3]])
if checkbox5: 
    initial_centroids=np.array([[0,0],[0.5,0], [1,0]])




# User choose number of iterations (limit to 6)
max_iter=5;
max_iter=st.slider('Iterations ',0,14)
K=3

st.set_option('deprecation.showPyplotGlobalUse', False)
def plotProgress(X,centroids,K,max_iter):
    plt.figure(figsize=(6,10))
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 6})
    assigned_centroid=np.zeros((X.shape[0],centroids.shape[1]))
    
    for iter in range(0,max_iter):
        idx=closestCentroids(centroids,X)
        if iter==0:
            pass
        else: 
            centroids=computeCentroids(X,idx,K)
        for index,i in enumerate(idx):
            assigned_centroid[index]=centroids[int(idx[i])]
        error=np.sqrt(np.sum((X-assigned_centroid)**2))
        plt.subplot(5,3,iter+1)
        plt.tight_layout()
        sns.scatterplot(x=sample_x[:,0],y=sample_x[:,1],hue=idx.reshape(300,),s=10,alpha=0.8);
        plt.scatter(x=centroids[:,0],y=centroids[:,1],color='red',s=10)
        plt.xticks(np.arange(0,10,1))
        plt.yticks(np.arange(0,7,1))
        plt.title('Iteration '+str(iter)+' Error: '+str(round(error,2)))
        plt.legend([],[], frameon=False)
        
        
plotProgress(sample_x,initial_centroids,K,max_iter)
st.pyplot()
 
 # Analyzing error



st.header('Application: Reducing the number of colors in an image')
st.write('''Using K Means to find K colours that best groups all the colors in the image.

Image data: Made out of many tiny pixels each with one uniform colour. A 180 x210 image has 180 rows and 210 columns of
pixels. Every color can be represented by a combination of the Red, Blue, Green scale at different brightness. The brightness
of an image ranges from 0-255. Combining all these elements, this image can be represented by a 180 x 210 x 3 3D matrix. 

The memory required to store 256 possible values is 8 bits, an image with 180 x 210 x 3 dimension will require 907,200 bits of storage. 
Representing 8 possible values requires 3 bits, and the same image will only need 340,200 bits storage. 

The objective of clustering is then to find the K number of colors that best represent the image and compress the image. 
''')

# 1.1 Reading and processing image data 
A=plt.imread('imag4.jpg')
A=A/255
X=np.reshape(A,(A.shape[0]*A.shape[1],A.shape[2]))
st.title('Compressing Image with k Means Cluster')

# 1.2 User selection 
K=st.selectbox('Number of colors, K',list(range(3,20)))
max_iter=st.selectbox('number of iterations',list(range(1,20)))

# 1.3 Runnig the algorithm
centroids,idx=runKMeans(X,K,max_iter)
X_recovered=centroids[idx,:]
X_recovered=np.reshape(X_recovered,(A.shape))

# 1.4 Return image result
st.write('Original image')
st.image(A)
if st.button(f'Get compressed image with {K} colors'):
    st.write('Compressed image')
    st.image(X_recovered)


st.header('Simple implementation with sklearn')
st.write('''Implement the algorithm with pre-written codes with the sklearn library. The model itself can be implemented
in 1 line of code! Here, it is important to tune and choose the correct parameter for an effective algorithm. The usual 
data cleaning, formatting and scaling is crucial. ''')
st.code('''
from sklearn.cluster import KMeans

# Defining the model and 
kmeans=KMeans(n_clusters=4, max_iter=30, random_state=42)

# Training the model with our data
kmeans.fit(X)

# Accessing results from the model
kmeans.cluster_centers_  # final centroids
kmeans.labels_           # Each datapoint's group
kmeans.inertia_          # Total squared distances of all datapoints and their respective centroids

''')



