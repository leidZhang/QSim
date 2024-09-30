class AnomalousEpisodeException(Exception):
    """
    AnomalousEpisodeException is an exception that is raised 
    when the episode is anomalous.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)