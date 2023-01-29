import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#DEFININDO MATRIZ DE CONFUSÃO

def plot_confusion_matrix(y_true, y_pred,classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    #classes = classes[unique_labels(y_true, y_pred)]
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",fontsize=16,
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


x=pd.read_csv(r'C:\Users\Kaue\Desktop\Dados simulação\Banco de dados 27-10-10\Todas correntes resumidas\Resumo 25 11 19\IP 5s Resumido com stats.csv',index_col=0, sep=',').T
 
x = x.to_numpy()
y = np.ones((2100), dtype=np.int32)
y[   :300] = 100      
y[301:600] = 2    
y[601:900] = 3     
y[901:1200] = 4        
y[1201:1500] = 97      
y[1501:1800] = 102     
y[1801:] = 105

scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
x = scaler.fit_transform(x)

x = PCA(0.99).fit_transform(x)
PCA1 = len(x[0])

lda = LinearDiscriminantAnalysis(n_components=6)
LDA1 = lda.n_components
x = lda.fit(x, y).transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3) #DIVIDE DADOS
del x, y

param_grid = {'C': [10,1,1e3, 5e3, 1e4, 5e4, 1e5],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1,1,10], }

clf = GridSearchCV(SVC(kernel='rbf'),param_grid, cv=5, iid=False)
clf = clf.fit(X_train, y_train)

acertos = accuracy_score(y_test, clf.predict(X_test), normalize=False, sample_weight=None)
T_test = len(y_test)
Acuracia = (acertos/T_test)*100

C = clf.best_estimator_.C
gamma = clf.best_estimator_.gamma

class_names = ['100% Mass','2 graus','3 graus','4 graus','97% Mass','102% Mass','105% Mass']
plot_confusion_matrix(y_test, clf.predict(X_test),classes = class_names,title='Confusion matrix')
t1 = ("Kernel =  RBF")
t2 = ("C =  "+str(C))
t3 = ("Gamma =  "+str(gamma))
t4 = ("Teste =  "+str(T_test))
plt.text(6.5, 7, t1, fontsize=11, style='oblique', ha='left', va='top', wrap=True)
plt.text(6.5, 7.2, t2, fontsize=11, style='oblique', ha='left', va='top', wrap=True)
plt.text(6.5, 7.4, t3, fontsize=11, style='oblique', ha='left', va='top', wrap=True)
plt.text(6.5, 7.6, t4, fontsize=11, style='oblique', ha='left', va='top', wrap=True)

plt.rcParams['figure.figsize'] = (9,7)
#plt.tight_layout()
plt.tick_params(axis = 'y', labelsize = 11)
plt.tick_params(axis = 'x', labelsize = 11)
plt.savefig(str(Acuracia)+' IP - SVM - C '+str(C)+'- G '+str(gamma)+'- RBF 22h.png', dpi = 470)