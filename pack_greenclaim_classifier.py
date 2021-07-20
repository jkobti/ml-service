from bento_flair_service import greenClaimClassifier
from flair.models import TextClassifier

model = TextClassifier.load("./models/greenclaim_classifier.pt")

greenClaimService = greenClaimClassifier()

greenClaimService.pack("model", model)
saved_path = greenClaimService.save()
print(saved_path)