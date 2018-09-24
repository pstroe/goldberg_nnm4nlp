
# coding: utf-8

# # Multi-Class Classification - Language Classification
# 
# This notebook implements the method presented in Goldberg's [2017] book "Neural Network Methods for Natural Language Processing". It shows the steps you need to go through in order to successfully train a classifier, and it should also, so I hope, illustrate the notational differences between Goldberg and standard machine learning literature.
# 
# $NOTE$: There is no cross-validation etc. to find optimal parameters. This is simply to show how multi-class classification works. This will be part of a tutorial session and all other concepts will be explained there.
# 
# Author: Phillip Ströbel

# ## Getting and cleaning the data
# 
# The data consists of downloaded Wikipedia articles (see `urls.txt`) in German, English, French, Spanish, Italian and Finnish (instead of "O" in Goldberg). The data is in HTML, so we need to some preprocessing to get the text out of it. We also restrict ourselfes to the characters from a to z in the alphabet (as described in Goldberg). In this fashion, we get rid of all the Umlauts (ä, ö, ü) and all other characters with diacritics (as, e.g., the é or ç in French). Note however, that if these characters ocurring in bigrams would probably be good features. In some way, we still keep the information "special character" by not fully deleting the character, but by replacing it by the dollar sign "\$". Furthermore, we replace all punctuation marks and digits by dollar signs as well. As such, all special characters, digits, and punctuation marks are mapped to $. The space will be replaced by an underscore "\_". We then represent each langauge by 28 characters, as is suggested by Goldberg.

# ### Cleaning HTML
# We first strip the HTML to get only the text of the Wikipedia page.

# #### Get the html files

# In[2]:


import re
import numpy as np
from bs4 import BeautifulSoup
from urllib.request import urlopen
from collections import defaultdict

seed = np.random.seed(seed=200)  # set a seed for random, so results are reproducible

article_dict = defaultdict(lambda: defaultdict(str))

regex = r'[\n ]{2,}'
pattern = re.compile(regex)

urls = open('urls.txt', 'r').readlines()

for index, url in enumerate(urls):
    language = url[8:10]
    doc_id = 'doc_%d' % index
    html = urlopen(url.strip()).read()    
    soup = BeautifulSoup(html, 'html.parser')
    raw = soup.body.get_text()  # only get text from the text body (this excludes headers and should exclude navigation bars)
    raw = re.sub(pattern, ' ', raw)  # replace multiple breaks and spaces by only one space
    raw = re.sub(r'\n', ' ', raw)  # replace every line break with a space
    article_dict[language][doc_id] = raw.lower()  # assign each text to its language and lower all uppercase characters


# ### Preprocessing --> prepare the text
# replace special characters and digits

# In[3]:


preprocessed_dict = defaultdict(lambda: defaultdict(str))

abc = r'[a-z]'
abc_pattern = re.compile(abc)

for lang, doc in article_dict.items():
    for doc, text in doc.items():
        for char in text:
            if re.match(abc_pattern, char):
                preprocessed_dict[lang][doc] += char
            elif re.match(' ', char):
                preprocessed_dict[lang][doc] += '_'
            else:
                preprocessed_dict[lang][doc] += '$'


# ### Count bigrams --> Feature extraction

# The distribution of bigrams will be our only feature. We could extend this by taking into account other n-grams.

# In[4]:


charset = 'abcdefghijklmnopqrstuvwxyz$_'  # define the character set we want to use


# In[5]:


from itertools import combinations_with_replacement, permutations

def bigrams(text):
    """
    Function to extract bigrams from text and calculate their distribution
    :param text: text string
    :return: dictionary containing bigrams as keys, and the normalised count as values
    """
    combs = combinations_with_replacement(charset, 2)
    perms = permutations(charset, 2)
    bigram_dict = dict()
    
    for comb in set(list(combs) + list(perms)):
        bigram_dict[''.join(comb)] = 0
        
    doc_length = len(text)
    
    for index in range(0, len(text)-1):
        bigram = text[index] + text[index+1]
        bigram_dict[bigram] += 1
                
    for bigram, count in bigram_dict.items():
        bigram_dict[bigram] = count/doc_length

    return bigram_dict              


# ### Put data into pandas dataframe
# The pandas dataframe allows us to conveniently represent all the data we need in one table. So let's do this. But first we need to extract the features.

# In[6]:


bigram_dict_full = defaultdict(lambda: defaultdict(dict))

for lang, doc in preprocessed_dict.items():
    for doc, text in sorted(doc.items()):
        bigram_dict = bigrams(text)
        bigram_dict_full[lang][doc] = bigram_dict


# In[7]:


import pandas as pd

col_names = ['y'] + sorted(bigram_dict_full['en']['doc_0'].keys())
my_df = dict()

for col in col_names:
    my_df[col] = list()
    
df = pd.DataFrame(my_df)

for lang, doc in bigram_dict_full.items():
    for key, value in doc.items():
        df_obj = value
        df_obj['y'] = lang
        df = df.append(df_obj, ignore_index=True)
        
df.head()
        


# In[8]:


df.shape


# Now we split the data into the label vector \begin{equation}\mathbf{y}\end{equation} and a training data matrix \begin{equation}\mathbf{X}\end{equation}. But first, we shuffle the df and split it into a training and a test set.

# Moreover, it is necessary for many machine learning tasks to standardise the data. Our aim is for each feature to be represented by a column vector in which values have zero mean and unit variance.

# In[9]:


def normalise_point(datapoint, mean, std):
    """
    normalise a datapoint to zero mean and unit variance.
    :param datapoint: value as a float
    :param mean: mean of data vector x
    :param std: standard deviation of data vector x
    :return: normalised datapoint (float)
    """
    return (datapoint - mean)/std

def normalise_matrix(matrix):
    """
    normalises the data matrix
    :param matrix: input matrix
    :return: normalised matrix
    """
    train_normalised = matrix.copy()
    
    for col in matrix:
        try:
            mean = matrix[col].mean()
            std = matrix[col].std()
            for index, item in enumerate(matrix[col]):
                train_normalised.loc[index, col] = normalise_point(item, mean, std)
        except ZeroDivisionError:
            train_normalised.loc[index, col] = 0.0
        except TypeError:
            continue
    return train_normalised


# In[10]:


df_norm = normalise_matrix(df)


# Split the data into a train set and a test set

# In[11]:


train = df_norm.sample(frac=0.9, random_state=seed)
test = df_norm.drop(train.index)


# Define the different sets

# In[12]:


# for training
y_train = train.y
X_train = train.drop('y', axis=1)

# for testing
y_test = test.y
X_test = test.drop('y', axis=1)


# Check the shapes

# In[13]:


print('Training samples shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test samples shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


# We should binarise our labels, although libraries like sklearn can also deal with non-numeric data

# In[14]:


from sklearn import preprocessing

lb = preprocessing.LabelBinarizer()
lb.fit(['en', 'fr', 'de', 'it', 'es', 'fi'])


# In[15]:


lb.classes_


# We do this for both our training and test labels:

# In[16]:


y_train = lb.transform(y_train)
y_test = lb.transform(y_test)


# Labels are now one-hot encoded:

# In[17]:


y_train[0]


# We almost have everything now. However, we need to take care of the bias and the weight matrix. The hypothesis ŷ is given by:
# \begin{equation}
# \mathbf{\hat{y}}=\mathbf{x}\cdot\mathbf{W}+\mathbf{b}
# \end{equation}
# We can achieve this by appending 1 to each feature vector x, and the whole weight vector b to the weight matrix W. This is called the bias trick. Note that the dimensions of X_train change, and that the weight matrix W will have match the dimensions (same number of rows as X has columns).

# In[18]:


bias_vector = np.ones([X_train.shape[0], 1])
X_train['bias'] = bias_vector
X_train


# In[19]:


# initialise weight matrix with small weights

np.random.seed(seed=200)

W = np.random.randn(X_train.shape[1], len(lb.classes_)) * 0.0001
#W = np.zeros([X.shape[1], len(lb.classes_)])


# We see that the dimensions are right. The dot product of a specific row from X_train and the weight matrix W constitutes a forward pass and calculates the score for each class.

# In[20]:


W.shape


# In[21]:


X_train.shape


# In[22]:


X_train[5:6].dot(W)


# We see that the values for the highest score of the dot product is not the score of the true label. Our aim is to change this by implementing a support vector classifier.

# In[23]:


X_train[5:6].dot(W).max(axis=1)


# In[24]:


X_train[5:6].dot(W)[y_train[5:6].argmax()]


# Important: we follow kind of a naive implementation. The aim is to be able to understand what is going on!
# 
# In order to quantify how good (or how bad) our weight matrix W can predict the data in our training set, we need to implement a loss function. Here we take a go at the hinge loss, which tries to predict the correct class with a margin of at least one to all other classes (or in this case, like presented in Goldberg, to the class which does not equal the true class, but which scores highest). In my understanding, this is a one-vs-one approach (true class vs. class with highest score (but doesn't equal the true class)).

# In[116]:


def hinge_loss(x, y, W, index):
    """
    Calculates the loss of a single data point by taking the prediction of the correct value and the the prediction of
    the value of next highest score, following Crammer and Singer (2001)
    :param x: sample point x as a vector
    :param y: correct label y for x as a vector
    :param W: weight matrix
    :param index: column index of data matrix X
    :return: loss
    """
    loss = 0
    y_index = y[index].argmax()
    y_value = x.dot(W)[y_index]
    y_hat_max_value = np.delete(x.dot(W), y_index).max()
    #for j in range(0, y.shape[1]):  # in case we wanted to classify against all other classes (one-vs-all) --> currently one-vs-one
        #if j == y_index:
            #continue
    loss += max(0, 1 - (y_value - y_hat_max_value))
    return loss


# With matrix multiplication, we could get all the scores at once. In the following, however, we focus on an approach which takes sample by sample and calculates the loss and the gradients.

# In[26]:


scores = X_train.dot(W)  # simple matrix multiplication to get all scores


# In[27]:


scores


# In[120]:


def gradient(X, y, W):
    """
    compute the gradient
    :param X: data matrix (train) 
    :param y: the corresponding 
    :param W: weight matrix
    :return: loss and Jacobian dW with all gradients
    """
    dW = np.zeros(W.shape)
    
    total_loss = 0.0
    
    for index, x in enumerate(X.as_matrix()):
        y_index = y[index].argmax()
        y_value = x.dot(W)[y_index]
        y_hat_max_value = np.delete(x.dot(W), y_index).max()
        loss = max(0, 1 - (y_value - y_hat_max_value))
        total_loss += loss
        y_hat_max_index = np.delete(x.dot(W), y_index).argmax() + 1
        if loss > 0:  # not sure whether we need this if statement
            dW[:, y_hat_max_index] += x.transpose()
            dW[:, y_index] -= x.transpose()
            
    return total_loss, dW
    


# In[121]:


def gradient_descent(X, y, W, eta, steps):
    """
    Perform gradient descent for a number of times with a fixed learning rate eta
    :param X: data matrix
    :param y: labels
    :param W: weight matrix
    :param eta: learning rate
    :param steps: number of times gradient descent should be performed
    :return: learned representation matrix W_learned
    """
    W_learned = W.copy()
    
    for step in range(0, steps):
        loss, dW = gradient(X, y, W_learned)
        print(loss)
        W_learned = W_learned - eta * dW
        
    return W_learned
    


# In[122]:


W_star = gradient_descent(X_train, y_train, W, eta=0.001, steps=10)


# ### Testing
# Let's test if our learned representation of the data is any good at classifying the data in the test set. Of course we need the bias in our test set as well!

# In[83]:


bias_vector_test = np.ones([X_test.shape[0], 1])
X_test['bias'] = bias_vector_test


# In[84]:


for index, x in enumerate(X_test.dot(W_star).as_matrix()):
    pred = x.argmax()
    true_label = y_test[index].argmax()
    print(pred, true_label)


# Not too bad! But Goldberg mentioned something about regularisation, so we should take this into account as well!

# In[113]:


def gradient_reg(X, y, W, lam):
    """
    compute the gradient
    :param X: data matrix (train) 
    :param y: the corresponding 
    :param W: weight matrix
    :param lam: reguliser lambda
    :return: Jacobian dW with all gradients
    """
    dW = np.zeros(W.shape)
    
    total_loss = 0.0
    
    for index, x in enumerate(X.as_matrix()):
        y_index = y[index].argmax()
        y_value = x.dot(W)[y_index]
        y_hat_max_value = np.delete(x.dot(W), y_index).max()
        loss = max(0, 1 - (y_value - y_hat_max_value)) + lam * np.linalg.norm(W, 2)
        total_loss += loss
        y_hat_max_index = np.delete(x.dot(W), y_index).argmax() + 1
        if loss > 0:  # not sure whether we need this if statement
            dW[:, y_hat_max_index] += x.transpose()
            dW[:, y_index] -= x.transpose()
        
    dW += 2 * lam * W
            
    return total_loss, dW

def gradient_descent_reg(X, y, W, eta, steps):
    """
    Perform gradient descent for a number of times with a fixed learning rate eta
    :param X: data matrix
    :param y: labels
    :param W: weight matrix
    :param eta: learning rate
    :param steps: number of times gradient descent should be performed
    :return: learned representation matrix W_learned
    """
    W_learned = W.copy()
    
    for step in range(0, steps):
        loss, dW = gradient_reg(X, y, W_learned, 10)
        print(loss)
        W_learned = W_learned - eta * dW
        
    return W_learned
    


# In[114]:


W_star_reg = gradient_descent_reg(X_train, y_train, W, eta=0.001, steps=10)


# In[115]:


for index, x in enumerate(X_test.dot(W_star_reg).as_matrix()):
    pred = x.argmax()
    true_label = y_test[index].argmax()
    print(pred, true_label)


# If we look at the two different weight matrices (one regularised, the other not), we notice the following:

# In[109]:


W_star[0:5]


# In[110]:


W_star_reg[0:5]


# ## By the way ...
# ### In scikit-learn it's much easier to implement this :-)
# 

# In[39]:


from sklearn.svm import LinearSVC
clf = LinearSVC(random_state=0, multi_class='crammer_singer', loss='hinge')
clf.fit(X_train, train.y)


# In[40]:


clf.predict(X_test)


# In[41]:


test.y


# We see that with our naive implementation, we do not much worse than with scikit's. scikit's implementation is of course much more elaborate and uses the vectorised operation and possibly other optimisation techniques in order to make its SVM (or SVC) better.
