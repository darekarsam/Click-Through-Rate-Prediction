from pathlib import Path
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle

def oneHotEncoding(features, uniqueValues):
	for filename in Path.cwd().glob("*.pickle"):
		with open(filename, 'rb') as handle:
			df = pickle.load(handle)
		
		for col in uniqueValues:
			# import ipdb; ipdb.set_trace()
			# print(str(filename).split('/')[-1])
			# print(col)
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

def partialFit():
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

	# import ipdb; ipdb.set_trace()
	from sklearn import metrics
	from sklearn import cross_validation
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.cross_validation import StratifiedKFold
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.linear_model import LogisticRegression
	from sklearn.svm import SVC

	Y = df.click	
	X = df.drop('click', axis=1).as_matrix()
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
	scoring = 'accuracy'

	for name, model in models:
		print('evaluating '+ name)
		kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
		cv_results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(msg)

	from matplotlib import pyplot as plt
	# Creating boxplot for Model comparison
	fig = plt.figure()
	fig.suptitle('Model Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	plt.show()
	import ipdb; ipdb.set_trace()
		

def main():
	# dataFolder = Path.cwd().joinpath('Data')
	# preProcessing(dataFolder)
	partialFit()

if __name__ == "__main__":	
	main()