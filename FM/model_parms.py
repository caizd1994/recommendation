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

  # optimizer
  parser.add_argument('--optimizer', default='adam', choices=['adam', 'adagrad'])
  parser.add_argument('--lr', type=float, default=0.001)

  #necessary dir
  parser.add_argument('--save_model_dir', default='save_model')
  parser.add_argument('--training_path', default="D:\\Game\\recommendation\data\\train.tfrecords")
  parser.add_argument('--validation_path', default="D:\\Game\\recommendation\\data\\train.tfrecords")
  parser.add_argument('--feature_dict', default=["uid", "city","item_id","author_id","item_city"])

  #task
  parser.add_argument('--task', default="finish")

  args = parser.parse_args()
  return args
