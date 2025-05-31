import joblib


class LoadedClassificationModel:
    def __init__(self, meta_model_path):
        self.meta_model_path = meta_model_path
        self.x_test = None

    def predict(self, x_test):
        self.x_test = x_test
        try:
            meta_model = joblib.load(self.meta_model_path)
        except FileNotFoundError as e:
            print(f"Error loading model or mlb: {e}")
            print("Please ensure the files exist at the specified paths.")
            # You might want to exit or handle this error appropriately
            exit()

        results = meta_model.predict([self.x_test])
        return results
