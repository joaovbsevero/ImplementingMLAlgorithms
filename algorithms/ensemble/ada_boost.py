from ..algorithm import Algorithm


class AdaBoost(Algorithm):
    @staticmethod
    def name():
        return "ada"

    @staticmethod
    def description() -> str:
        return "Combine many weak machine-learning models to create a powerful classification model"

    @staticmethod
    def run():
        pass
