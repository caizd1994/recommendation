from FM.FMModel import FM_model
from common_utils.utils import *
from FM.model_parms import init_model_args

feature_list = ["uid", "city", "item_id", "author_id", "item_city"]
bucket_size = [73974, 396, 4122689, 850308, 461]
# feature_list = ["city", "item_id", "author_id", "item_city"]
# bucket_size = [396, 4122689, 850308, 461]


class RecommendModelHandler(object):
    def __init__(self, train_dataset_path, val_dataset_path, save_model_dir,
                 num_epochs, batch_size, embedding_size, learning_rate,
                 optimizer, task, num_threads=1):
        self._num_threads = num_threads
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._embedding_size = embedding_size
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._train_dataset_path = train_dataset_path
        self._val_dataset_path = val_dataset_path
        self._save_model_dir = save_model_dir
        self._task = task


    def build_feature(self):
        first_feature_columns = []
        second_feature_columns = []
        for feat, size in zip(feature_list, bucket_size):
            cate_id = tf.feature_column.categorical_column_with_identity(feat, size)
            first_feature_columns.append(tf.feature_column.embedding_column(cate_id, dimension=1))
            second_feature_columns.append(tf.feature_column.embedding_column(cate_id, dimension=self._embedding_size))
        print("Number of feature column features:%d" % (len(first_feature_columns)))
        return first_feature_columns, second_feature_columns

    def build_model(self):
        """ build recommend model framework"""
        first_feature_columns, second_feature_columns = self.build_feature()

        config = tf.estimator.RunConfig().replace(
            session_config=tf.ConfigProto(device_count={'CPU': self._num_threads}),
            log_step_count_steps=20)
        params = {
            'learning_rate': self._learning_rate,
            'optimizer': self._optimizer,
            'first_feature_columns': first_feature_columns,
            'second_feature_columns': second_feature_columns
        }

        model = tf.estimator.Estimator(
            model_fn=FM_model.train_op,
            model_dir=self._save_model_dir,
            params=params,
            config=config)
        return model

    def train(self):
        """
        Train model
        """
        parms = init_model_args()
        print("here",parms.training_path)
        model = self.build_model()
        train_spec = tf.estimator\
            .TrainSpec(input_fn=lambda: input_fn(filenames=parms.training_path
                                                 ,num_epochs=parms.num_epochs
                                                 ,batch_size=parms.batch_size
                                                 ,features_dict=feature_list))
        val_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(filenames=parms.training_path
                                                                   ,num_epochs=parms.num_epochs
                                                                   ,batch_size=parms.batch_size
                                                                   ,features_dict=feature_list), steps=1)
        tf.estimator.train_and_evaluate(model, train_spec, val_spec)
        # model.evaluate(input_fn=lambda: self.input_fn(data_mode='train'))
        # pred = model.predict(input_fn=lambda: self.input_fn(data_mode='val'))
