import argparse

def init_model_args():
  """
  Basic but important params for traning
  """
  parser = argparse.ArgumentParser()
  # training size
  parser.add_argument('--batch_size', type=int, default=40)
  parser.add_argument('--embedding_size', type=int, default=40)
  parser.add_argument('--num_epochs', type=int, default=200)
  parser.add_argument('--num_cross_layers', type=int, default=3)
  parser.add_argument('--hidden_units',  default=[200,128,128])


  #regularization
  parser.add_argument("--is_bn", default="false")
  parser.add_argument("--is_dropout", default="false")
  parser.add_argument("--dropout_rate", default=0.1)

  # optimizer
  parser.add_argument('--optimizer', default='adam', choices=['adam', 'adagrad'])
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument("--is_decy",default="true")
  parser.add_argument("--decay_steps", default=5000)
  parser.add_argument("--decay_rate", default=0.95)


  #necessary dir
  parser.add_argument('--save_model_dir', default='save_model')
  parser.add_argument('--training_path', default="F:\\project\\recommendation\data\\train.tfrecords")
  parser.add_argument('--validation_path', default="F:\\project\\recommendation\data\\train.tfrecords")
  parser.add_argument('--feature_dict', default=["uid", "city","item_id","author_id","item_city"])

  #task
  parser.add_argument('--task', default="finish")

  args = parser.parse_args()
  return args