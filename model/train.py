import tensorflow as tf

import model.model as model
import data.load as load


def train_summary(sess, sr_model):
    input_image_summary = tf.summary.image('input_image', sr_model.image, max_outputs=5)
    label_image_summary = tf.summary.image('label_image', sr_model.label, max_outputs=5)
    output_image_summary1 = tf.summary.image('output_image', sr_model.inference, max_outputs=5)
    output_image_summary2 = tf.summary.image('feature_image', sr_model.feature, max_outputs=5)
    output_image_summary3 = tf.summary.image('ilr_image', sr_model.ilr, max_outputs=5)

    # lr_sr_res_image_summary = tf.summary.image('residual(sr-lr)', sr_model.inference - sr_model.image, max_outputs=10)
    # hr_sr_res_image_summary = tf.summary.image('residual(hr-sr)', sr_model.label - sr_model.inference, max_outputs=10)

    learning_rate_hist = tf.summary.scalar('learning_rate', sr_model.learning_rate)
    loss_hist1 = tf.summary.scalar('loss/loss', sr_model.loss)
    loss_hist2 = tf.summary.scalar('loss/mse', sr_model.mse)
    loss_hist3 = tf.summary.scalar('loss/reg', sr_model.regularizer)
    psnr_hist1 = tf.summary.scalar('psnr', sr_model.psnr)

    merged = tf.summary.merge_all()
    writter = tf.summary.FileWriter('./summary/' + sr_model.model_config.model_name, sess.graph)

    return merged, writter


def train_loop(sess, sr_model, config, batch_len, images, labels):
    merged, writter = train_summary(sess, sr_model)
    step = 0

    for epoch in range(config.epochs):
        print('epoch : %d' % epoch)

        for batch_index in range(batch_len):

            # mini batch index
            start_ind = batch_index * config.batch_size
            end_ind = (batch_index + 1) * config.batch_size

            # mini batch
            batch_images = images[start_ind: end_ind]
            batch_labels = labels[start_ind: end_ind]

            # optimize the model
            sess.run(sr_model.train, feed_dict={
                sr_model.image: batch_images, sr_model.label: batch_labels
            })

            loss, mse, psnr = sess.run([sr_model.loss, sr_model.mse, sr_model.psnr], feed_dict={
                sr_model.image: batch_images, sr_model.label: batch_labels
            })

            # write the log for training
            summary = sess.run(merged, feed_dict={
                sr_model.image: batch_images, sr_model.label: batch_labels
            })

            step += 1

            if step % 100 == 0:
                print('\t-step : [%3d], loss : [%.8f], mse : [%.08f], psnr : [%.2f]' % (
                    step, loss, mse, psnr))

            if step % 100 == 0:
                writter.add_summary(summary, step)

        # save the trained mode for each epoch
        sr_model.save(sess, epoch)


def trainer(config):

    print('load dataset')
    print(config.dataset_name)
    images, labels = load.load_h5(config.dataset_name)

    print(images.shape)
    print(labels.shape)

    print('dataset is loaded')
    image_len = len(images)
    batch_len = image_len // config.batch_size + 1

    print("data size(len) : %d" % image_len)
    print("batch len : %d" % batch_len)

    with tf.Session() as sess:
        sr_model = model.Model(config)

        if config.pretrain:
            sess.run(tf.global_variables_initializer())
            sr_model.load(sess)
        else:
            sess.run(tf.global_variables_initializer())

        train_loop(sess, sr_model, config, batch_len, images, labels)
