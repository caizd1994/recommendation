import os
import tensorflow as tf
from FM.model_parms import init_model_args
from FM.model import RecommendModelHandler


def main():
    # basic logging setup for tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.INFO)

    # init necessary args
    args = init_model_args()

    train_dataset_path_list = []
    val_dataset_path_list = []
    # for i in range(0, 20000000, 4000000):
    #     path = "D:\\Game\\recommendation\\data\\train.tfrecords"
    #     train_dataset_path_list.append(path)
    #     val_dataset_path_list.append(path
    train_dataset_path_list = args.training_path
    val_dataset_path_list = args.validation_path

    print("training path list: {}".format(train_dataset_path_list))
    print("training path list: {}".format(val_dataset_path_list))

    save_model_dir = args.save_model_dir
    print("saving model in ... {}".format(save_model_dir))

    optimizer = args.optimizer
    learning_rate = args.lr
    print("we use {} as optimizer".format(optimizer))
    print("learning rate is set as  {}".format(learning_rate))

    batch_size = args.batch_size
    embedding_size = args.embedding_size
    num_epochs = args.num_epochs
    print("batch size: {}".format(batch_size))
    print("embedding size: {}".format(embedding_size))

    task = args.task
    print("task: {}".format(task))

    model = RecommendModelHandler(
        train_dataset_path=train_dataset_path_list,
        val_dataset_path=val_dataset_path_list,
        save_model_dir=save_model_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        embedding_size=embedding_size,
        learning_rate=args.lr,
        task=task)

    model.train()


if __name__ == '__main__':
    main()
