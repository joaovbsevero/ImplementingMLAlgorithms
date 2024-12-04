class Algorithm:
    @staticmethod
    def name() -> str:
        raise NotImplementedError

    @staticmethod
    def description() -> str:
        return ""

    @staticmethod
    def run():
        raise NotImplementedError
