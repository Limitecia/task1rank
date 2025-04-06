from src.service_codalab import CodalabSemeval2024T3Service


s = "uh ... delicates . And that would be your bras ... and your underpanty things ."

x = CodalabSemeval2024T3Service.normalize_utterance(s)
print(x)
