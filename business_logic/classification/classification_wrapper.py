from business_logic.classification.multilabel_classifier import predict


class LoadedClassificationModel:
    def __init__(self, save_dir, model_paths):
        self.save_dir = save_dir
        self.x_test = None

    def predict(self, x_test):
        self.x_test = x_test

        results = predict([self.x_test], )
        return results
