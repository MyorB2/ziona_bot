class LoadedClassificationModel:
    def __init__(self, model):
        self.model = model
        self.x_test = None

    def predict(self, x_test):
        self.x_test = x_test

        results = self.model.predict([self.x_test], )
        return results
