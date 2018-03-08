import pickle
import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np
import tensorflow as tf
import gc
import seaborn as sns
import matplotlib.pyplot as plt


def loadFile(num):
	with open(str(num) + ".pickle", 'rb') as handle:
		df = pickle.load(handle)
	cols = df.columns.tolist()
	cols.insert(len(cols)-1, cols.pop(cols.index('click')))
	return df[cols]

def getHeatmap(data):
	""" Function to get Heatmap of normalized Numerical values before 
		applying to the classifier
	"""
	corr = data.corr()
	mask = np.zeros_like(corr, dtype=np.bool)
	mask[np.triu_indices_from(mask)] = True
	f, ax = plt.subplots(figsize=(11, 9))
	cmap = sns.diverging_palette(220, 10, as_cmap=True)
	# import ipdb; ipdb.set_trace()
	ax = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, annot=True,
			square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
	ax.set_title("Heatmap of normalized continous variables")
	plt.savefig('corrHeatmap.png')
	print("Correlation Heatmap saved .. ")

def getTrainData():
	finalDf = pd.DataFrame()
	print("getting train data ...")

	#reusing the preprocessd data saved as pickle
	for i in range(7): #no 8 is test
		df = loadFile(i+1)

		minorDf = df[df.click == 1]

		#undersampling of majority class and keeping ratio 1:10
		majorDf = df[df.click == 0].sample(n=len(minorDf)*10, replace=False)

		dfSampled = pd.concat([minorDf, majorDf], axis=0)

		finalDf = finalDf.append(dfSampled)

	finalDf = finalDf.sample(frac=1)#shuffle dataframe
	
	# tempDf = finalDf[['width', 'height', 'slotPrice', 'bidprice', 'payPrice']]
	# getHeatmap(tempDf)

	Y = finalDf.click
	X = finalDf.drop('click', axis=1)

	del finalDf
	gc.collect()
	
	#Generate synthetic samples using SMOTE
	X, Y = SMOTE(random_state=7, ratio='auto').fit_sample(X,Y)
	Y = pd.get_dummies(pd.Series(Y)).as_matrix()
	return X,Y

def getTestData():
	print("getting test data ...")
	df = loadFile(8)
	Y = pd.get_dummies(df.click).as_matrix()
	X = df.drop('click', axis=1).as_matrix()
	return X,Y

def neural_network_model(data, keep_prob):
#initialize weights and bias having 0 mean randomly 
	seed = 149
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([features, n_nodes_hl1], seed=seed)),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1], seed=seed))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2], seed=seed)),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl2], seed=seed))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3], seed=seed)),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl3], seed=seed))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes], seed=seed)),
					'biases':tf.Variable(tf.random_normal([n_classes], seed=seed))}
	
	

	#forward propogation using relu activation
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)
	l1 = tf.nn.dropout(l1, keep_prob)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)
	l1 = tf.nn.dropout(l1, keep_prob)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)
	l1 = tf.nn.dropout(l1, keep_prob)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output



def train_neural_network(x, keep_prob):
	prediction = neural_network_model(x, keep_prob)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, 
		labels=y))
	
	optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
	hm_epochs = 100
	displayStep = 5
	init = tf.global_variables_initializer()

	epochValues = {
					'epoch' : [],
					'loss' : [],
					'trainAccuracy' : [],
					'trainPrecision' : [],
					'trainRecall' : [],

					'valAccuracy' : [],
					'valPrecision' : [],
					'valRecall' : [],

					'testAccuracy' : [],
					'testPrecision' : [],
					'testRecall' : []
	}
	
	predicted = tf.argmax(prediction, 1)
	actual = tf.argmax(y, 1)

	correct = tf.equal(predicted, actual)
	accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

	TP = tf.count_nonzero(predicted * actual, dtype=tf.float32)
	TN = tf.count_nonzero((predicted - 1) * (actual - 1), dtype=tf.float32)
	FP = tf.count_nonzero(predicted * (actual - 1), dtype=tf.float32)
	FN = tf.count_nonzero((predicted - 1) * actual, dtype=tf.float32)
	precision = tf.divide(TP, tf.add(TP, FP))
	recall = tf.divide(TP, tf.add(TP, FN))
	test_x, test_y = getTestData()

	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(hm_epochs):
			epoch_loss = 0
			i = 0
			while i < len(train_x):
				start = i
				end = i + batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])
				_, c = sess.run([optimizer, cost], 
								feed_dict={
									x: batch_x, 
									y: batch_y,
									keep_prob: 0.8
									})
				epoch_loss += c
				i += batch_size
			
			# if (epoch + 1) % displayStep == 0:
			print('Epoch {}/{} Loss: {:.4f}'.format(epoch+1, hm_epochs, epoch_loss))
			epochValues['valPrecision'] = list(np.asarray(epochValues['valPrecision'])*0.5)
			epochValues['trainPrecision'] = list(np.asarray(epochValues['trainPrecision'])*0.5)
			# epochValues['testPrecision'] = list(np.asarray(epochValues['valPrecision'])*0.25)

			epochValues['epoch'].append(epoch + 1)
			epochValues['loss'].append(epoch_loss)

			epochValues['valAccuracy'].append(accuracy.eval({x:val_x, y:val_y, keep_prob: 1.0}))
			epochValues['valPrecision'].append(precision.eval({x:val_x, y:val_y, keep_prob: 1.0}))
			epochValues['valRecall'].append(recall.eval({x:val_x, y:val_y, keep_prob: 1.0}))

			epochValues['trainAccuracy'].append(accuracy.eval({x:train_x, y:train_y, keep_prob: 1.0}))
			epochValues['trainPrecision'].append(precision.eval({x:train_x, y:train_y, keep_prob: 1.0}))
			epochValues['trainRecall'].append(recall.eval({x:train_x, y:train_y, keep_prob: 1.0}))

			epochValues['testAccuracy'].append(accuracy.eval({x:test_x, y:test_y, keep_prob: 1.0}))
			epochValues['testPrecision'].append(precision.eval({x:test_x, y:test_y, keep_prob: 1.0}))
			epochValues['testRecall'].append(recall.eval({x:test_x, y:test_y, keep_prob: 1.0}))

		print('Final Training Accuracy: ',accuracy.eval({x:train_x, y:train_y, keep_prob: 1.0}))
		print('Final Training Precision: ',precision.eval({x:train_x, y:train_y, keep_prob: 1.0}))
		print('Final Training Recall: ',recall.eval({x:train_x, y:train_y, keep_prob: 1.0}))
		print(" ")

		print('Final Validation Accuracy: ',accuracy.eval({x:val_x, y:val_y, keep_prob: 1.0}))
		print('Final Validation Precision: ',precision.eval({x:val_x, y:val_y, keep_prob: 1.0}))
		print('Final Validation Recall: ',recall.eval({x:val_x, y:val_y, keep_prob: 1.0}))
		print(" ")

		
		print('Final Test Accuracy: ',accuracy.eval({x:test_x, y:test_y, keep_prob: 1.0}))
		print('Final Test Precision: ',precision.eval({x:test_x, y:test_y, keep_prob: 1.0}))
		print('Final Test Recall: ',recall.eval({x:test_x, y:test_y, keep_prob: 1.0}))
		print(" ")

		# # Normalize Loss values for visualization purposes
		# lossSum = sum(epochValues['loss'])
		# epochValues['loss'] = [i / lossSum for i in epochValues['loss']]

		f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
		ax1.plot(epochValues['epoch'], epochValues['loss'], label="Loss", color='r')
		ax1.set_title("Loss")

		l1, = ax2.plot(epochValues['epoch'], epochValues['trainAccuracy'], label="Train Accuracy", color='b')
		l2, = ax2.plot(epochValues['epoch'], epochValues['testAccuracy'], label="Test Accuracy", color='g')
		l3, = ax2.plot(epochValues['epoch'], epochValues['valAccuracy'], label="Validation Accuracy", color='m')
		ax2.set_title("Accuracy")

		ax3.plot(epochValues['epoch'], epochValues['trainPrecision'], label="Train Precision", color='b')
		ax3.plot(epochValues['epoch'], epochValues['testPrecision'], label="Test Precision", color='g')
		ax3.plot(epochValues['epoch'], epochValues['valPrecision'], label="Validation Precision", color='m')
		ax3.set_title("Precision(train and val values are divided by 2 for )")

		ax4.plot(epochValues['epoch'], epochValues['trainRecall'], label="Train Recall", color='b')
		ax4.plot(epochValues['epoch'], epochValues['testRecall'], label="Test Recall", color='g')
		ax4.plot(epochValues['epoch'], epochValues['valRecall'], label="Validation Recall", color='m')
		ax4.set_title("Recall")

		plt.legend([l1, l2, l3],["Train", "Test", "Validation"])
		plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
		
		plt.savefig('timeframe.png')
		# plt.show()

		proba1 = prediction[:, 1]
		proba1 = proba1.eval({x:train_x, y:train_y, keep_prob: 1.0})
		index1 = actual.eval({x:train_x, y:train_y, keep_prob: 1.0})
		probaDf = pd.DataFrame({'proba1': proba1, 'click' : index1})

		f, ax = plt.subplots()
		ax = sns.distplot(probaDf[probaDf.click==0].proba1, hist=False, label="No Click", color='red')
		ax = sns.distplot(probaDf[probaDf.click==1].proba1, hist=False, label="Click", color='blue')
		ax.set_title("Score Distribution of Prediction by class variable")
		plt.savefig('distr.png')
		# plt.show()

		


X, Y = getTrainData()

# Shuffle arrays
s = np.arange(X.shape[0])
np.random.shuffle(s)
X = X[s]
Y = Y[s]

# Seperate training and validation set
split_size = int(X.shape[0]*0.8)
train_x, val_x = X[:split_size], X[split_size:]
train_y, val_y = Y[:split_size], Y[split_size:]

	# Neural Network size parameters
n_nodes_hl1 = 500
n_nodes_hl2 = 250
n_nodes_hl3 = 40
n_classes = 2
features = train_x.shape[1] 
batch_size = 256

x = tf.placeholder('float', [None, features])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder("float")
train_neural_network(x, keep_prob)

