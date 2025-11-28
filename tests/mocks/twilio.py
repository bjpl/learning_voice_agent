"""
Mock implementation of twilio library for testing

Provides minimal mock classes needed for tests to import successfully.
"""

from typing import Any, Dict, Optional
from unittest.mock import MagicMock


# Mock TwiML module
class twiml:
    """Mock twiml module"""

    class voice_response:
        """Mock voice_response module"""

        class VoiceResponse:
            """Mock VoiceResponse"""

            def __init__(self):
                self.content = []

            def say(self, message, **kwargs):
                """Mock say"""
                self.content.append({'say': message})
                return self

            def gather(self, **kwargs):
                """Mock gather"""
                gather_obj = Gather(**kwargs)
                self.content.append({'gather': gather_obj})
                return gather_obj

            def redirect(self, url, **kwargs):
                """Mock redirect"""
                self.content.append({'redirect': url})
                return self

            def hangup(self):
                """Mock hangup"""
                self.content.append({'hangup': True})
                return self

            def __str__(self):
                """Convert to TwiML XML string"""
                return '<Response></Response>'


        class Gather:
            """Mock Gather"""

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.content = []

            def say(self, message, **kwargs):
                """Mock say"""
                self.content.append({'say': message})
                return self


        class Say:
            """Mock Say"""

            def __init__(self, message, **kwargs):
                self.message = message
                self.kwargs = kwargs


# Expose at module level for direct imports
VoiceResponse = twiml.voice_response.VoiceResponse
Gather = twiml.voice_response.Gather
Say = twiml.voice_response.Say


class TwilioException(Exception):
    """Base Twilio exception"""
    pass


class TwilioRestException(TwilioException):
    """Mock Twilio REST exception"""

    def __init__(self, status: int, uri: str, msg: str = "", code: int = None, method: str = "GET"):
        self.status = status
        self.uri = uri
        self.msg = msg
        self.code = code
        self.method = method
        super().__init__(f"{status}: {msg}")


class Message:
    """Mock Twilio Message"""

    def __init__(self, sid: str = "SM123", status: str = "sent", **kwargs):
        self.sid = sid
        self.status = status
        for key, value in kwargs.items():
            setattr(self, key, value)


class Call:
    """Mock Twilio Call"""

    def __init__(self, sid: str = "CA123", status: str = "completed", **kwargs):
        self.sid = sid
        self.status = status
        for key, value in kwargs.items():
            setattr(self, key, value)


class Recording:
    """Mock Twilio Recording"""

    def __init__(self, sid: str = "RE123", **kwargs):
        self.sid = sid
        for key, value in kwargs.items():
            setattr(self, key, value)


class MessageList:
    """Mock message list"""

    def create(self, to: str, from_: str, body: str = "", media_url: list = None, **kwargs):
        """Mock message creation"""
        return Message(to=to, from_=from_, body=body)

    def list(self, **kwargs):
        """Mock list messages"""
        return []


class CallList:
    """Mock call list"""

    def create(self, to: str, from_: str, url: str = "", **kwargs):
        """Mock call creation"""
        return Call(to=to, from_=from_, url=url)

    def list(self, **kwargs):
        """Mock list calls"""
        return []


class RecordingList:
    """Mock recording list"""

    def list(self, **kwargs):
        """Mock list recordings"""
        return []


class Api:
    """Mock Twilio API"""

    def __init__(self):
        self.messages = MessageList()
        self.calls = CallList()
        self.recordings = RecordingList()


class RequestValidator:
    """Mock Twilio RequestValidator for signature validation"""

    def __init__(self, auth_token: str):
        self.auth_token = auth_token

    def validate(self, url: str, params: dict, signature: str) -> bool:
        """Mock validate method - always returns True in tests"""
        return True

    def compute_signature(self, url: str, params: dict) -> str:
        """Mock compute_signature"""
        return "mock_signature"


class Client:
    """Mock Twilio Client"""

    def __init__(self, account_sid: str = None, auth_token: str = None):
        self.account_sid = account_sid or "AC123"
        self.auth_token = auth_token or "test_token"
        self.messages = MessageList()
        self.calls = CallList()
        self.recordings = RecordingList()
        self.api = Api()


# Mock REST module
class Rest:
    """Mock REST module"""
    Client = Client


rest = Rest()

# Mock request_validator module
class request_validator:
    """Mock request_validator module"""
    RequestValidator = RequestValidator
