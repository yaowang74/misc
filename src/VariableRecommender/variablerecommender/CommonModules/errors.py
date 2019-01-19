# -*- coding: utf-8 -*-
"""
User defined errors

@author: H211803
"""
"""Predefined generic errors that are raised when application execution fails.

"""

OK = 'OK'
CANCELLED = 'CANCELLED'
UNKNOWN = 'UNKNOWN'
INVALID_ARGUMENT = 'INVALID_ARGUMENT'
NOT_SUPPORTED = 'NOT_SUPPORTED'
ALREADY_EXISTS = 'ALREADY_EXISTS'
PERMISSION_DENIED = 'PERMISSION_DENIED'
UNAUTHENTICATED = 'UNAUTHENTICATED'
INTERNAL = 'INTERNAL'
UNAVAILABLE = 'UNAVAILABLE'
NO_DATA_FOUND = 'NO_DATA_FOUND'
LOADING_DATA_FAILED = 'LOADING_DATA_FAILED'

NO_SECTION_FOUND = 'NO_SECTION_FOUND'
NO_OPTION_FOUND = 'NO_OPTION_FOUND'
NOT_FOUND = 'NOT_FOUND'
NO_CONFIG_PARAM = 'CONFIG_PARAM_NOT_FOUND'
INVALID_CONFIG_VALUE = 'INVALID_CONFIG_VALUE'


class Error(Exception):
    """Base class for exceptions.

    """

    def __init__(self, message, error_code):
        super(Exception, self).__init__()
        self._message = message
        self._error_code = error_code

    @property
    def message(self):
        return self._message

    @property
    def error_code(self):
        return self._error_code


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class NotFoundError(Error):
    """Raised when a requested entity or resource
    (e.g., a file or directory or configuration ) was not found.
    """
    def __init__(self, message):
        """Create a `NotFoundError`."""
        super(NotFoundError, self).__init__(message, NOT_FOUND)
