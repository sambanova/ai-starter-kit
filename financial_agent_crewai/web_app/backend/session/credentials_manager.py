from typing import Dict, Optional

from cryptography.fernet import Fernet


class APIKeyManager:
    def __init__(self, encryption_key: Optional[str] = None) -> None:
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)

    def encrypt_keys(self, keys: Dict[str, str]) -> bytes:
        """
        Encrypts a dictionary of API keys using the configured encryption key.

        Args:
            keys: A dictionary containing API keys (key-value pairs) to be encrypted.

        Returns:
            bytes: The encrypted version of the API keys, encoded as bytes.

        Raises:
            ValueError: If the keys cannot be properly converted to a string for encryption.
        """
        return self.cipher_suite.encrypt(str(keys).encode())

    def decrypt_keys(self, encrypted_keys: bytes) -> Dict[str, str]:
        """
        Decrypts the encrypted API keys back into their original dictionary format.

        Args:
            encrypted_keys: The encrypted API keys as bytes to be decrypted.

        Returns:
            The decrypted dictionary of API keys.

        Raises:
            ValueError: If the decryption fails or the decrypted data cannot be parsed into a dictionary.
        """
        decrypted = self.cipher_suite.decrypt(encrypted_keys).decode()
        eval_decrypted: Dict[str, str] = eval(decrypted)
        return eval_decrypted
