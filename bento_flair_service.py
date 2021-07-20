from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from flair_model_artifact import FlairModelArtifact
from flair.data import Sentence


@env(infer_pip_packages=True)
@artifacts([FlairModelArtifact("model")])
class greenClaimClassifier(BentoService):
    @api(input=DataframeInput(), batch=True)
    def predict(self, df):
        sentences = [Sentence(row['clean_tweet']) for index, row in df.iterrows()]
        self.artifacts.model.predict(sentences, verbose = False, mini_batch_size=32)
        predictions=[]
        for sentence in sentences:
            if sentence.labels:
                predictions.append(sentence.labels[0].value)
            else:
                predictions.append("no_label")
        return predictions