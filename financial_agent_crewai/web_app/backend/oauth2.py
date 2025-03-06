import datetime
import os
from typing import Any, Dict

import jwt
from fastapi.exceptions import HTTPException
from jwt.exceptions import InvalidTokenError

from financial_agent_crewai.web_app.backend import schemas

SECRET_KEY = os.getenv('SECRET_KEY')
ALGORITHM = os.getenv('ALGORITHM')
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES'))


def create_access_token(data: Dict[str, str]) -> Any:
    to_encode = data.copy()

    expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({'exp': expire})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


def verify_access_token(token: str, credentials_exception: HTTPException) -> schemas.TokenData:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        session_token: str = str(payload.get('session_token'))
        if session_token is None:
            raise credentials_exception
        token_data = schemas.TokenData(session_token=session_token)
    except InvalidTokenError:
        raise credentials_exception

    return token_data
