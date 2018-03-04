from pathlib import Path
import pandas as pd
import pickle
import numpy as np
from matplotlib import pyplot as plt

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


def oneHotEncoding(features, uniqueValues):
	for filename in Path.cwd().glob("*.pickle"):
		with open(filename, 'rb') as handle:
			df = pickle.load(handle)
		
		for col in uniqueValues:
			cat = df[col].astype('category', categories=uniqueValues[col])
			tempdf = pd.get_dummies(cat, prefix=col, drop_first=True)

			#concatenating new columns
			df = pd.concat([df, tempdf], axis=1)

			#dropping original column
			df.drop([col], axis=1, inplace=True) 
		# import ipdb; ipdb.set_trace()
		with open(filename, 'wb') as handle:
			pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
		print('File saved as {}..'.format( str(filename).split('/')[-1] ))

def preProcessing(dataFolder):

	columnHeaders = ['bidID', 'Timestamp', 'logType', 'XYZID', 'useragent', 'ip', \
		'region', 'city', 'adexchange', 'domain', 'url', 'anonURLID',\
		'adSlotID', 'width', 'height', 'visibility', 'format', 'slotPrice',\
		'creativeId', 'bidprice', 'payPrice', 'keypageUrl', 'adverId',\
		 'userTag']                  

	implogs = sorted(list(dataFolder.glob('imp.201310[1-2][9,0,1,2,3,4,5,6,7].txt.bz2')))
	clklogs = sorted(list(dataFolder.glob('clk.201310[1-2][9,0,1,2,3,4,5,6,7].txt.bz2')))

	features = {
		'categorical' : ['region', 'city', 'adexchange','visibility'],
		'numeric' : ['width', 'height', 'slotPrice', 'bidprice', 'payPrice'],
		'drop' : ['bidID','Timestamp', 'logType', 'XYZID', 'useragent', 'ip', 'domain', 'url',\
				'anonURLID', 'adSlotID', 'creativeId', 'userTag','keypageUrl', 'format', 'bidID','adverId']
	}


	#dict to get unique values in all categorical variables
	uniqueValues = {x:[] for x in features['categorical']}

	#Preprocessing:
	#drop features and normalize numerical features and store as pickle file
	for i in range(8):
		print('Preprocessing file {}..'.format(str(implogs[i]).split('/')[-1]))
		clkDf = pd.read_csv(clklogs[i], sep='\t', names=columnHeaders, compression='bz2')
		impDf = pd.read_csv(implogs[i], sep='\t', names=columnHeaders, compression='bz2')

		#normalize numerial features
		tempdf = impDf[features['numeric']]
		tempdf = MinMaxScaler().fit_transform(tempdf)
		tempdf = pd.DataFrame(tempdf, columns=features['numeric'])
		
		impDf[features['numeric']] = tempdf #replace normalized columns

		#introducing target variable
		impDf['click'] = 0
		#make click column=1 if it is in click DF
		impDf.loc[impDf.bidID.isin(clkDf.bidID),'click'] = 1

		impDf.index = impDf.bidID #change index of dataframe
		impDf.drop(features['drop'], axis=1, inplace=True)

		# import ipdb; ipdb.set_trace()
		#todo: check
		# replacing NaN values with -1
		impDf.fillna(value=0, inplace=True)

		#storing the file as pickle so that can be loaded faster next time
		with open(str(i+1)+'.pickle', 'wb') as handle:
			pickle.dump(impDf, handle, protocol=pickle.HIGHEST_PROTOCOL)
		print('File saved as {}.pickle..'.format(i+1))
		

		for col in features['categorical']:
			uniqueValues[col] = sorted(list(set(uniqueValues[col] + list(impDf[col].unique()))))
		# import ipdb; ipdb.set_trace()

	oneHotEncoding(features, uniqueValues)

def basicModels(X, Y):
	# prepare configuration for cross validation test with basic classifier
	num_folds = 5
	num_instances = len(X)
	seed = 7
	print('preparing models')
	models = []
	models.append(('LR', LogisticRegression()))
	# models.append(('KNN', KNeighborsClassifier()))
	models.append(('DT', DecisionTreeClassifier()))
	models.append(('NB', MultinomialNB()))
	models.append(('SVM', SVC()))
	# evaluate each model
	results = []
	names = []

	for name, model in models:
		print('evaluating '+ name)
		kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
		cv_results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring='precision')
		# import ipdb; ipdb.set_trace()
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(msg)

	
	# Creating boxplot for Model comparison
	fig = plt.figure()
	fig.suptitle('Model Comparison using Precision scores')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	plt.show()

def runRandomForest(X, Y):
	num_folds = 5
	num_instances = len(X)
	seed = 7
	model = RandomForestClassifier(n_estimators=500, max_depth=4, n_jobs=1)
	print("Training Random forest model")
	model.fit(X, Y)
	with open("model.pickle", 'wb') as handle:
		pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print("model saved")

	# with open("model.pickle", 'rb') as handle:
	# 	model = pickle.load(handle)

	importances = model.feature_importances_
	std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
	indices = np.argsort(importances)[::-1]
	# Print the feature ranking
	print("Feature ranking:")

	for f in range(20):
	    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

	import ipdb;ipdb.set_trace()
	# Plot the feature importances of the forest
	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(20), importances[indices][:20], color="r", yerr=std[indices][:20], align="center")
	plt.xticks(range(20), indices[:20])
	plt.xlim([-1, 20])
	plt.show()

	predicted = cross_validation.cross_val_predict(model, X, Y, cv=5)
	print("classification_report:")
	print(metrics.classification_report(Y, predicted))

	Y = pd.Series(Y, name='Actual')
	Y = Y.reset_index(drop=True)
	predicted = pd.Series(predicted, name='Predicted')

	print("test set confusion_matrix:")
	print(pd.crosstab(Y, predicted))
	return indices[:20]

def runModels():
	finalDf = pd.DataFrame()
	for i in range(7): #no 8 is test
		with open(str(i+1) + ".pickle", 'rb') as handle:
			df = pickle.load(handle)
		
		cols = df.columns.tolist()
		cols.insert(len(cols)-1, cols.pop(cols.index('click')))
		df = df[cols]


		minorDf = df[df.click == 1]
		#undersampling of majority class and keeping ratio 1:10
		majorDf = df[df.click == 0].sample(n=len(minorDf)*10, replace=False)
		
		dfSampled = pd.concat([minorDf, majorDf], axis=0)

		finalDf = finalDf.append(dfSampled)

	#reshuffle the dataframe rows
	finalDf = finalDf.sample(frac=1)

	Y = df.click	
	X = df.drop('click', axis=1)

	X, Y = SMOTE(random_state=7, ratio = 1.0).fit_sample(X,Y)
	basicModels(X,Y)

	indexes = runRandomForest(X, Y)
	print("top 20 Important Features")
	print(finalDf.columns[indexes])

def runTest():
	with open("8.pickle", 'rb') as handle:
		testDf = pickle.load(handle)
	cols = testDf.columns.tolist()
	cols.insert(len(cols)-1, cols.pop(cols.index('click')))
	testDf = testDf[cols]
	yTest = testDf.click
	xTest = testDf.drop('click', axis=1)
	xTest = xTest.as_matrix()
	with open("model.pickle", 'rb') as handle:
		model = pickle.load(handle)
	yPredicted = model.predict(xTest)
	import ipdb; ipdb.set_trace()
	yTest = pd.Series(yTest, name='Actual')
	yTest = yTest.reset_index(drop=True)
	yPredicted = pd.Series(yPredicted, name='Predicted')

	print("test set confusion_matrix:")
	print(pd.crosstab(yTest, yPredicted))
	import ipdb;ipdb.set_trace()
	print("test set classification_report:")
	print(metrics.classification_report(yTest.tolist(), yPredicted.tolist()))

def main():
	dataFolder = Path.cwd().joinpath('Data')
	preProcessing(dataFolder)
	runModels()
	runTest()

if __name__ == "__main__":	
	main()