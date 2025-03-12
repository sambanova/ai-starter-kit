from typing import Any


class SnsdkWrapperRepositoryrError(Exception):
    """Base class for all exceptions raised by the SnsdkWrapperRepositoryrError."""

    def __init__(self, message: str, *args: Any) -> None:
        super().__init__(message, *args)
        self.message = message


class AppFetchError(SnsdkWrapperRepositoryrError):
    """Exception raised when fetching available apps fails."""

    def __init__(self, message: str = 'Error fetching available apps.', *args: Any) -> None:
        super().__init__(message, *args)


class DatasetFetchError(SnsdkWrapperRepositoryrError):
    """Exception raised when fetching datasets fails."""

    def __init__(self, message: str = 'Error fetching datasets.', *args: Any) -> None:
        super().__init__(message, *args)


class DatasetCreateError(SnsdkWrapperRepositoryrError):
    """Exception raised when creating a dataset fails."""

    def __init__(self, dataset_name: str, message: str = 'Error creating dataset.', *args: Any) -> None:
        super().__init__(message, *args)
        self.dataset_name = dataset_name


class DatasetDeleteError(SnsdkWrapperRepositoryrError):
    """Exception raised when deleting a dataset fails."""

    def __init__(self, dataset_name: str, message: str = 'Error deleting dataset.', *args: Any) -> None:
        super().__init__(message, *args)
        self.dataset_name = dataset_name
