import os
import tensorflow as tf
from deepfm.config import init_model_args
from deepfm.model import RecommendModelHandler


def main():
    # basic logging setup for tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.INFO)

    # init necessary args
    args = init_model_args()

    train_dataset_path_list = []
    val_dataset_path_list = []
    for i in range(0, 20000000, 4000000):
        path = "F:\\project\\recommendation\data\\train.tfrecords"
        train_dataset_path_list.append(path)
        val_dataset_path_list.append(path)
    print("training path list: {}".format(train_dataset_path_list))
    print("training path list: {}".format(val_dataset_path_list))

    save_model_dir = args.save_model_dir
    print("saving model in ... {}".format(save_model_dir))

    optimizer = args.optimizer
    learning_rate = args.lr
    print("we use {} as optimizer".format(optimizer))
    print("learning rate is set as  {}".format(learning_rate))

    batch_size = args.batch_size
    print("batch size: {}".format(batch_size))
    embedding_size = args.embedding_size
    print("embedding size: {}".format(embedding_size))
    num_epochs = args.num_epochs


    task = args.task
    print("task: {}".format(task))

    is_decy = args.is_decy
    print("is_decy: {}".format(is_decy))

    decay_steps = args.decay_steps
    print("decay_steps: {}".format(decay_steps))

    decay_rate = args.decay_rate
    print("decay_rate: {}".format(decay_rate))

    is_bn = args.is_bn
    print("is_bn: {}".format(is_bn))

    is_dropout = args.is_dropout
    print("is_dropout: {}".format(is_dropout))

    dropout_rate = args.dropout_rate
    print("dropout_rate: {}".format(dropout_rate))

    hidden_units = args.hidden_units
    print("hidden_units: {}".format(hidden_units))

    model = RecommendModelHandler(
        train_dataset_path=train_dataset_path_list,
        val_dataset_path=val_dataset_path_list,
        save_model_dir=save_model_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        embedding_size=embedding_size,
        learning_rate=args.lr,
        task=task,
        is_decy=is_decy,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        is_bn=is_bn,
        is_dropout=is_dropout,
        dropout_rate=dropout_rate,
        hidden_units=hidden_units
    )

    model.train()


if __name__ == '__main__':
    main()
