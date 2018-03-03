from pathlib import Path
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier

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
	paths = sorted(list(Path.cwd().glob("*.pickle")))
	for i in range(8):
		with open(str(i+1) + ".pickle", 'rb') as handle:
			df = pickle.load(handle)
		Y = df.click
		X = df.drop('click', axis=1)
		import ipdb; ipdb.set_trace()
		model = SGDClassifier()
		model.partial_fit(X,Y, classes=Y.unique())


		

def main():
	# dataFolder = Path.cwd().joinpath('Data')
	# preProcessing(dataFolder)
	partialFit()

if __name__ == "__main__":	
	main()