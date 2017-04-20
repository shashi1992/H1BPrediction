import tensorflow as tf
from six.moves import cPickle as pickle
import numpy as np

with open('seperatedData.pickle', 'rb') as file:
	pickleData = pickle.load(file)
	trainingData, trainingLabel = pickleData['trainingData'], pickleData['trainingLabel']
	validData, validLabel = pickleData['validData'], pickleData['validLabel']
	testData, testLabel = pickleData['testData'], pickleData['testLabel']

batch_size = 128
feature_size = 6
num_labels = 2

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

keep_prob = 0.6

graph = tf.Graph()
with graph.as_default(): 
	tf_trainingData = tf.placeholder(tf.float32, shape=(batch_size, feature_size))
	tf_trainingLabel = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_validData = tf.constant(validData)
	tf_validLabel = tf.constant(validLabel)
	tf_testData = tf.constant(testData)

	layer1_weights = tf.Variable(tf.truncated_normal([feature_size, 1024], stddev=0.1))
	layer1_biases = tf.Variable(tf.zeros([1024]))
	layer2_weights = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.1))
	layer2_biases = tf.Variable(tf.zeros([512]))
	layer3_weights = tf.Variable(tf.truncated_normal([512, 256], stddev=0.1)) 
	layer3_biases = tf.Variable(tf.zeros([256]))
	layer4_weights = tf.Variable(tf.truncated_normal([256, 128], stddev=0.1))
	layer4_biases = tf.Variable(tf.zeros([128]))
	layer5_weights = tf.Variable(tf.truncated_normal([128, num_labels], stddev=0.1))
	layer5_biases = tf.Variable(tf.zeros([num_labels]))

	def model(data): 
		dataflow = tf.matmul(data, layer1_weights) + layer1_biases
		nonlinear = tf.nn.dropout(tf.nn.relu(dataflow), keep_prob = keep_prob)
		dataflow = tf.matmul(nonlinear, layer2_weights) + layer2_biases
		nonlinear = tf.nn.dropout(tf.nn.relu(dataflow), keep_prob = keep_prob)
		dataflow = tf.matmul(nonlinear, layer3_weights) + layer3_biases 
		nonlinear = tf.nn.dropout(tf.nn.relu(dataflow), keep_prob = keep_prob)
		dataflow = tf.matmul(nonlinear, layer4_weights) + layer4_biases 
		nonlinear = tf.nn.dropout(tf.nn.relu(dataflow), keep_prob = keep_prob)
		return tf.matmul(nonlinear, layer5_weights) + layer5_biases
	
	globalStep = tf.Variable(0)
	learning = tf.placeholder("float")
	#learning_rate = tf.train.exponential_decay(learning, globalStep, 100000, 0.9, staircase=True)
	learning_rate = tf.train.exponential_decay(learning, globalStep, 100000, 0.9, staircase=True)
	logits = model(tf_trainingData)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = tf_trainingLabel))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(model(tf_validData))
	test_prediction = tf.nn.softmax(model(tf_testData))

num_steps = 233006 // 128

with tf.Session(graph = graph) as session:
	tf.global_variables_initializer().run()
	print('Initialized')
	for step in range(num_steps):
		offset = (step * batch_size) % (trainingLabel.shape[0] - batch_size)
		batch_data = trainingData[offset:(offset + batch_size), :]
		batch_labels = trainingLabel[offset:(offset + batch_size), :]
		feed_dict = {tf_trainingData: batch_data, tf_trainingLabel: batch_labels, learning: 0.0000005}

		_, l, predict = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)

		if step % 100 == 0:
			print('Minibatch loss at step %d: %f' % (step, l))
			print('Minibatch accuracy: %.1f%%' % accuracy(predict, batch_labels))
			print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), validLabel))
			print()
	print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), testLabel))

