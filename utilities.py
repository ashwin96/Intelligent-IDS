import sys
import numpy as np
from sklearn import tree,linear_model
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.plotly as py
import pandas as pd;
traning_data=sys.argv[1]
testing_data=sys.argv[2]
newdata = []
def get_data_details(csv_data):
        data=np.genfromtxt(csv_data, delimiter=",",usecols=(0,1,2,3))
        #data=pd.read_csv(csv_data, delimiter=',')
        features=data[:,[0,1,2]]
        labels=data[:,3]
        """for X in features:
                plt.plot(X,color="ro")
        for y in labels:
                plt.plot(y,color='ro')"""
        return features,labels

def get_occuracy(real_labels,predicted_labels,fltr):
        real_label_count=0.0
        predicted_label_count=0.0
        data = np.genfromtxt(traning_data, delimiter=',',usecols=(0,1,2,3))
        for real_label in real_labels:
                if real_label==fltr:
                        real_label_count+=1
        i=0;
        for predicted_label in predicted_labels:
                if predicted_label==fltr:
                        predicted_label_count+=1
                        str2 = str(i);
                        newdata.append(str2);
                        plt.plot(data[i,0],data[i,2],'bx');
                i+=1
        plt.show();
        print "Real number of attacks:"+str(real_label_count)
        print "Predicted number of attacks:"+str(predicted_label_count)

        precision=predicted_label_count*100/real_label_count
        return precision,newdata;
