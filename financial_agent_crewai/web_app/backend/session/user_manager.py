import uuid
from typing import Dict

import redis

from financial_agent_crewai.web_app.backend.session.credentials_manager import APIKeyManager
from financial_agent_crewai.src.exceptions import InvalidSessionError


class UserSessionManager:
    @staticmethod
    def create_session(
        user_id: str, api_keys: Dict[str, str], key_manager: APIKeyManager, redis_client: redis.Redis
    ) -> str:
        """
        Creates a secure session for a user by encrypting their API keys and storing them in Redis.

        Args:
            user_id: The unique identifier for the user.
            api_keys: A dictionary containing the user's API keys.
            key_manager: An instance of the APIKeyManager to handle encryption.
            redis_client: The Redis client used to store the session data.

        Returns:
            session_token: A session token that uniquely identifies the created session.
        """
        encrypted_keys = key_manager.encrypt_keys(api_keys)

        session_token = str(uuid.uuid4())

        redis_client.setex(f'session:{session_token}', 36000, encrypted_keys)

        return session_token

    @staticmethod
    def get_session_keys(session_token: str, key_manager: APIKeyManager, redis_client: redis.Redis) -> Dict[str, str]:
        """
        Retrieves and decrypts the API keys associated with a given session token.

        Args:
            session_token: The unique session token for the user session.
            key_manager: An instance of the APIKeyManager to handle decryption.
            redis_client: The Redis client used to retrieve the session data.

        Returns:
            The decrypted API keys associated with the session.

        Raises:
            InvalidSessionError: If the session is invalid or expired, or if the session data cannot be found.
        """

        encrypted_keys = redis_client.get(f'session:{session_token}')

        if not encrypted_keys:
            raise InvalidSessionError('Invalid or expired session')

        return key_manager.decrypt_keys(encrypted_keys)
