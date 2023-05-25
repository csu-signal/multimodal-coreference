import os
from collections import defaultdict

def get_ecbp_imagedict():
  images = defaultdict(list)

  directory = os.path.join(os.path.dirname(__file__), 'data/ECB+')

  for article_dir in os.listdir(directory):
      article_path = os.path.join(directory, article_dir)
      if os.path.isdir(article_path):
        for filename in os.listdir(article_path):
          file_end = filename.split(".")[-1]
          if file_end == "jpg":
            images[article_dir].append(os.path.join(article_path, filename))
  return images
