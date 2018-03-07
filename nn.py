import pickle
import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np
import tensorflow as tf
import gc
import seaborn as sns
import matplotlib.pyplot as plt


finalDf = pd.DataFrame()
print("getting data")
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

finalDf = finalDf.sample(frac=1)

Y = finalDf.click
X = finalDf.drop('click', axis=1)

del finalDf
gc.collect()


X, Y = SMOTE(random_state=7, ratio='auto').fit_sample(X,Y)

Y = pd.get_dummies(pd.Series(Y)).as_matrix()


# To stop potential randomness
seed = 128
split_size = int(X.shape[0]*0.8)
# Shuffle arrays
s = np.arange(X.shape[0])
np.random.shuffle(s)
X = X[s]
Y = Y[s]
# Y = finalDf.click

# import ipdb; ipdb.set_trace()

# Seperate training and validation setn

train_x, val_x = X[:split_size], X[split_size:]
train_y, val_y = Y[:split_size], Y[split_size:]

import tensorflow as tf
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

def neural_network_model(data, keep_prob):
#initialize weights and bias having 0 mean randomly 
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([features, n_nodes_hl1])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					'biases':tf.Variable(tf.random_normal([n_classes]))}
	
	

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
	# import ipdb; ipdb.set_trace()
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, 
		labels=y))
	
	optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost)
	hm_epochs = 5
	displayStep = 10
	init = tf.global_variables_initializer()

	epochValues=[]
	costValues=[]

	# Turn on interactive plotting
	# plt.ion()
	# # Create the main, super plot
	# fig = plt.figure()

	# # Create two subplots on their own axes and give titles
	# ax = plt.subplot()
	# ax.set_title("TRAINING LOSS", fontsize=18)
	# plt.tight_layout()

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
			epochValues.append(epoch + 1)
			costValues.append(epoch_loss)
			 # Write summary stats to writer
			# writer.add_summary(summary_results)
			# print(summary_results)
			print('Epoch {}/{} Loss: {}'.format(epoch+1, hm_epochs, epoch_loss))
		# costLine, = ax.plot(epochValues, costValues)
		# fig.canvas.draw()
		# 	# time.sleep(1)
		# import ipdb; ipdb.set_trace()
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

		# **
		print('Validation Accuracy: ',accuracy.eval({x:val_x, y:val_y, keep_prob: 1.0}))
		print('Validation Precision: ',precision.eval({x:val_x, y:val_y, keep_prob: 1.0}))
		print('Validation Recall: ',recall.eval({x:val_x, y:val_y, keep_prob: 1.0}))

		print('Training Accuracy: ',accuracy.eval({x:train_x, y:train_y, keep_prob: 1.0}))
		print('Training Precision: ',precision.eval({x:train_x, y:train_y, keep_prob: 1.0}))
		print('Training Recall: ',recall.eval({x:train_x, y:train_y, keep_prob: 1.0}))


train_neural_network(x, keep_prob)