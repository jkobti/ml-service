import os
import json
from bentoml.utils import cloudpickle
from bentoml.exceptions import InvalidArgument
from bentoml.service.artifacts import BentoServiceArtifact
from flair.models import TextClassifier


class FlairModelArtifact(BentoServiceArtifact):
    def __init__(self, name):
      print('__init__')
      super(FlairModelArtifact, self).__init__(name)
      self._model = None

    def pack(self, model, metadata=None):
      self._model = model
      return self

    def get(self):
      return self._model

    def save(self, dst):
      self._model.save(self._file_path(dst))

    def load(self, path):
      print(path)
      model = TextClassifier.load(self._file_path(path))
      return self.pack(model)

    def _file_path(self, base_path):
      print("_file_path")  
      return os.path.join(base_path, self.name + '.json')