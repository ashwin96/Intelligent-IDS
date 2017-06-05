import matplotlib
matplotlib.use('TkAgg')
from utilities import *
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from pylab import scatter, show, legend, xlabel, ylabel
from mpl_toolkits.mplot3d import Axes3D
from tkinter import *
import tkSimpleDialog
import tkMessageBox
#Get training features and labeles
def plotit(data):
	data = np.array(data)
	length = data.shape[0]
	width = data.shape[1]
	x,y= np.meshgrid(np.arange(length), np.arange(width))
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1, projection='3d')
	ax.scatter(x,y,1,s=20);
	plt.show();
print "Extracting training_features.........."
training_features,traning_labels=get_data_details(traning_data)
print training_features;
#print traning_labels;
plt.scatter(*zip(*training_features))
plt.show();
plotit(training_features);
#Get testing features and labels
print "Extracting testing features............"
testing_features,testing_labels=get_data_details(testing_data)
print testing_features;
print testing_labels;
plt.scatter(*zip(*testing_features))
plt.show();
plotit(testing_features);
### DECISON TREE CLASSIFIER
print "\n\n=-=-=-=-=-=-=- Decision Tree Classifier -=-=-=-=-=-=-=-\n"

#Instanciate the classifier
attack_classifier=tree.DecisionTreeClassifier()
#Train the classifier
attack_classifier=attack_classifier.fit(training_features,traning_labels)
tree.export_graphviz(attack_classifier,out_file='tree.jpg')  
#print attack_classifier;
#get predections for the testing data
predictions=attack_classifier.predict(testing_features)
newdata=[]
str1,newdata = get_occuracy(testing_labels,predictions,1)
print "The precision of the Decision Tree Classifier is:"+str(str1)+"%"
tkMessageBox.showinfo("Classfication","Classification complete"+'\n'+"Press Ok to Classification Accuracy.") 
tkMessageBox.showinfo("Accuracy of Decision Trees","Accuracy="+str(str1)+"%"+'\n'+"Press OK to Results");
root = Tk()
t = Text(root)
t.insert(END,"List of malicious clients"+'\n')
for x in newdata:
    t.insert(END, "Client "+str(x)+ '\n')
t.pack()
root.mainloop()
#root.mainloop();