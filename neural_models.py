from __future__ import division

from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import tensorflow as tf


def dan(train_examples, train_labels, test_examples, test_labels, num_labels, embedding_dim):
    train_mat = [ex[0] for ex in train_examples]
    stylo_train_mat = [ex[1] for ex in train_examples]
    test_mat = [ex[0] for ex in test_examples]
    stylo_test_mat = [ex[1] for ex in test_examples]

    inputs = tf.placeholder(tf.float32, embedding_dim)
    stylo_inputs = tf.placeholder(tf.float32, 32)
    full_inputs = tf.concat([inputs, stylo_inputs], 0)
    w = tf.get_variable('w', [num_labels, embedding_dim + 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    z = tf.tensordot(w, full_inputs, 1)
    probs = tf.nn.softmax(z)
    one_best = tf.argmax(probs)
    label = tf.placeholder(tf.int32, 1)
    label_onehot = tf.reshape(tf.one_hot(label, num_labels), shape=[num_labels])
    loss = tf.negative(tf.log(tf.tensordot(probs, label_onehot, 1)))

    global_step = tf.contrib.framework.get_or_create_global_step()
    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    num_epochs = 20

    with tf.Session() as sess:
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        for i in range(0, num_epochs):
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(train_examples)):
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                [_, loss_this_instance] = sess.run([train_op, loss], feed_dict={inputs: np.mean(train_mat[ex_idx], axis=0),
                                                                                stylo_inputs: stylo_train_mat[ex_idx],
                                                                                label: np.array([train_labels[ex_idx]])})
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)

            preds = []
            for ex_idx in xrange(len(test_examples)):
                [probs_this_instance, pred_this_instance] = sess.run([probs, one_best], feed_dict={inputs: np.mean(test_mat[ex_idx], axis=0),
                                                                                                   stylo_inputs: stylo_test_mat[ex_idx]})
                preds.append(pred_this_instance)

            print "Dev Accuracy: " + str(accuracy_score(test_labels, preds))
            print confusion_matrix(test_labels, preds)


def pad_to_length(np_arr, length):
    result = np.zeros((length, 300))
    less = length if length < len(np_arr) else len(np_arr)
    for i in xrange(less):
        result[i] = np_arr[i]
    return result


def generate_batches(examples, labels, batch_size):
    length = (len(examples) // batch_size) * batch_size
    batches = []
    for i in xrange(0, length, batch_size):
        batches.append((examples[i:i+batch_size], labels[i:i+batch_size]))
    return batches


def cnn(train_examples, train_labels, test_examples, test_labels, num_classes, embedding_size):
    seq_max_len = 20
    train_mat = np.array([pad_to_length(ex[0], seq_max_len) for ex in train_examples])
    train_labels_mat = np.array(train_labels)
    test_mat = np.array([pad_to_length(ex[0], seq_max_len) for ex in test_examples])
    stylo_mat = np.array([ex[1] for ex in train_examples])
    stylo_test_mat = np.array([ex[1] for ex in test_examples])

    # Hyperparams
    stylo_len = np.shape(stylo_mat)[1]
    num_epochs = 50
    batch_size = 10
    filter_widths = [3, 4, 5]
    filters_per_region = 100
    num_filters = filters_per_region * len(filter_widths)

    # Network
    stylo_inputs = tf.placeholder(tf.float32, [None, stylo_len])
    inputs = tf.placeholder(tf.float32, [None, seq_max_len, embedding_size])
    conv1_inputs = tf.expand_dims(inputs, -1)
    dropout_rate = tf.placeholder(tf.float32)

    feature_vector = []
    for f in filter_widths:
        filters = tf.get_variable("filters_%d" % f, [f, embedding_size, 1, filters_per_region], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        conv1 = tf.nn.conv2d(conv1_inputs, filters, [1, 1, 1, 1], 'VALID')
        bias = tf.get_variable("bias_%d" % f, [conv1.shape[-1]], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        biased_conv = tf.nn.relu(tf.nn.bias_add(conv1, bias))
        pool1 = tf.nn.max_pool(biased_conv, [1, seq_max_len-f+1, 1, 1], [1, 1, 1, 1], 'VALID')
        feature_vector.append(pool1)

    feature_vector = tf.concat(feature_vector, 3)
    feature_vector = tf.reshape(feature_vector, (-1, num_filters))
    feature_vector = tf.nn.dropout(feature_vector, dropout_rate)
    feature_vector = tf.concat([feature_vector, stylo_inputs], 1)

    W = tf.get_variable("W", [num_filters + stylo_len, num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    probs = tf.nn.softmax(tf.tensordot(feature_vector, W, 1))
    one_best = tf.argmax(probs, axis=1)

    # Input for the gold label so we can compute the loss
    label = tf.placeholder(tf.int32, [None])
    one_hot = tf.one_hot(label, num_classes)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot, logits=probs)
    global_step = tf.contrib.framework.get_or_create_global_step()
    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        tf.set_random_seed(0)
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            loss_this_iter = 0
            j = 0
            stylo_batches = generate_batches(stylo_mat, train_labels, batch_size)
            for example_batch, label_batch in generate_batches(train_mat, train_labels_mat, batch_size):
                [_, loss_this_instance] = sess.run([train_op, loss], feed_dict={inputs: example_batch,
                                                                                stylo_inputs: stylo_batches[j][0],
                                                                                label: label_batch,
                                                                                dropout_rate: 0.65})
                j += 1
                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)

            preds = []
            correct_r_3 = 0.0
            correct_r_5 = 0.0
            for ex_idx in xrange(len(test_examples)):
                [probs_this_instance, pred_this_instance] = sess.run([probs, one_best], feed_dict={inputs: [test_mat[ex_idx]],
                                                                                 stylo_inputs: [stylo_test_mat[ex_idx]],
                                                                                 dropout_rate: 1.0})
                preds.append(pred_this_instance[0])

                lps = [(i, probs_this_instance[0][i]) for i in xrange(len(probs_this_instance[0]))]
                slps = sorted(lps, key=lambda tup: tup[1], reverse=True)
                top_3 = [slps[t][0] for t in range(3)]
                top_5 = [slps[t][0] for t in range(5)]

                if test_labels[ex_idx] in top_3:
                    correct_r_3 += 1.0
                    correct_r_5 += 1.0
                elif test_labels[ex_idx] in top_5:
                    correct_r_5 += 1.0

            print "Dev Accuracy, Recall 1: " + str(accuracy_score(test_labels, preds))
            print "Dev Accuracy, Recall 3: " + str(correct_r_3/len(test_labels))
            print "Dev Accuracy, Recall 5: " + str(correct_r_5/len(test_labels))
            print confusion_matrix(test_labels, preds)


def lstm(train_examples, train_labels, test_examples, test_labels, num_classes, embedding_size):
    seq_max_len = 20
    train_mat = np.array([pad_to_length(ex[0], seq_max_len) for ex in train_examples])
    train_labels_mat = np.array(train_labels)
    test_mat = np.array([pad_to_length(ex[0], seq_max_len) for ex in test_examples])
    stylo_mat = np.array([ex[1] for ex in train_examples])
    stylo_test_mat = np.array([ex[1] for ex in test_examples])
    stylo_len = np.shape(stylo_mat)[1]

    # Hyperparams
    num_epochs = 50
    batch_size = 10
    hidden_size = 300

    # Network
    stylo_inputs = tf.placeholder(tf.float32, [None, stylo_len])
    inputs = tf.placeholder(tf.float32, [None, seq_max_len, embedding_size])
    seq_inputs = tf.split(inputs, seq_max_len, 1)
    seq_inputs = [tf.squeeze(t, 1) for t in seq_inputs]

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    b_size = tf.placeholder(tf.int32, [])
    init_state = lstm_cell.zero_state(b_size, tf.float32)

    outputs, state = tf.nn.static_rnn(lstm_cell, seq_inputs, dtype=tf.float32, initial_state=init_state)
    outputs = outputs[-1]

    W = tf.get_variable("W", [hidden_size + stylo_len, num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    hidden = tf.concat([outputs, stylo_inputs], 1)
    prediction = tf.tensordot(hidden, W, 1)
    probs = tf.nn.softmax(prediction)
    one_best = tf.argmax(probs, axis=1)

    # Input for the gold label so we can compute the loss
    label = tf.placeholder(tf.int32, [None])
    one_hot = tf.one_hot(label, num_classes)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot, logits=probs)
    global_step = tf.contrib.framework.get_or_create_global_step()
    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        tf.set_random_seed(0)
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            loss_this_iter = 0
            j = 0
            stylo_batches = generate_batches(stylo_mat, train_labels, batch_size)
            for example_batch, label_batch in generate_batches(train_mat, train_labels_mat, batch_size):
                [_, loss_this_instance] = sess.run([train_op, loss], feed_dict={inputs: example_batch,
                                                                                stylo_inputs: stylo_batches[j][0],
                                                                                label: label_batch,
                                                                                b_size: 10})
                j += 1
                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)

            preds = []
            correct_r_3 = 0.0
            correct_r_5 = 0.0
            for ex_idx in xrange(len(test_examples)):
                [probs_this_instance, pred_this_instance] = sess.run([probs, one_best], feed_dict={inputs: [test_mat[ex_idx]],
                                                                                 stylo_inputs: [stylo_test_mat[ex_idx]],
                                                                                 b_size: 1})
                preds.append(pred_this_instance[0])

                lps = [(i, probs_this_instance[0][i]) for i in xrange(len(probs_this_instance[0]))]
                slps = sorted(lps, key=lambda tup: tup[1], reverse=True)
                top_3 = [slps[t][0] for t in range(3)]
                top_5 = [slps[t][0] for t in range(5)]

                if test_labels[ex_idx] in top_3:
                    correct_r_3 += 1.0
                    correct_r_5 += 1.0
                elif test_labels[ex_idx] in top_5:
                    correct_r_5 += 1.0

            print "Dev Accuracy, Recall 1: " + str(accuracy_score(test_labels, preds))
            print "Dev Accuracy, Recall 3: " + str(correct_r_3/len(test_labels))
            print "Dev Accuracy, Recall 5: " + str(correct_r_5/len(test_labels))
            print confusion_matrix(test_labels, preds)
